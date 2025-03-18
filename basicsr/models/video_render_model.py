import torch
from collections import Counter
from os import path as osp
from torch import distributed as dist
from tqdm import tqdm
from collections import OrderedDict
from basicsr.metrics import calculate_metric
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.utils.dist_util import get_dist_info
from basicsr.utils.registry import MODEL_REGISTRY
from .video_base_model import VideoBaseModel
import numpy as np
from einops import rearrange
from basicsr.utils.img_process_util import USMSharp

@MODEL_REGISTRY.register()
class VideoRenderModel(VideoBaseModel):
    def __init__(self, opt):
        super(VideoRenderModel, self).__init__(opt)
        if self.is_train:
            self.fix_flow_iter = opt['train'].get('fix_flow')

        # self.lq_sharp = opt['train'].get('lq_sharp', False)
        # self.gt_sharp = opt['train'].get('gt_sharp', False)


        self.usm_sharpener = USMSharp().to(self.device)

    def feed_data(self, data):
        self.lq, self.gt = {}, {}
        get_data = lambda key: data[key].to(self.device) if key in data else None

        self.lq['Image'] = get_data('lq_Image')
        self.gt['Image'] = get_data('gt_Image')

        # if self.lq_sharp and self.is_train:
        #     self.lq['Image'] = self.usm_sharpener(self.lq['Image'])

        # if self.gt_sharp and self.is_train:
        #     self.gt['Image'] = self.usm_sharpener(self.gt['Image'])

        for image_type in ['Normal', 'BRDF', 'Depth', 'Emit', 'Motion']:
          self.lq[image_type] = get_data('lq_' + image_type)
          self.lq['gt_' + image_type] = get_data('gt_' + image_type)

    def setup_optimizers(self):
        train_opt = self.opt['train']
        flow_lr_mul = train_opt.get('flow_lr_mul', 1)
        logger = get_root_logger()
        logger.info(f'Multiple the learning rate for flow network with {flow_lr_mul}.')
        if flow_lr_mul == 1:
            optim_params = self.net_g.parameters()
        else:  # separate flow params and normal params for different lr
            normal_params = []
            flow_params = []
            for name, param in self.net_g.named_parameters():
                if 'spynet' in name:
                    flow_params.append(param)
                else:
                    normal_params.append(param)
            optim_params = [
                {  # add normal params first
                    'params': normal_params,
                    'lr': train_opt['optim_g']['lr']
                },
                {
                    'params': flow_params,
                    'lr': train_opt['optim_g']['lr'] * flow_lr_mul
                },
            ]

        optim_type = train_opt['optim_g'].pop('type')
        self.optimizer_g = self.get_optimizer(optim_type, optim_params, **train_opt['optim_g'])
        self.optimizers.append(self.optimizer_g)

    def optimize_parameters(self, current_iter):
        if self.fix_flow_iter:
            logger = get_root_logger()
            if current_iter == 1:
                logger.info(f'Fix flow network and feature extractor for {self.fix_flow_iter} iters.')
                for name, param in self.net_g.named_parameters():
                    if 'spynet' in name or 'edvr' in name:
                        param.requires_grad_(False)
            elif current_iter == self.fix_flow_iter:
                logger.warning('Train all the parameters.')
                self.net_g.requires_grad_(True)

        self.optimizer_g.zero_grad()
        self.output = self.net_g(self.lq)

        l_total = 0
        loss_dict = OrderedDict()
        # pixel loss
        if self.cri_pix:
            l_pix = self.cri_pix(self.output, self.gt['Image'])
            l_total += l_pix
            loss_dict['l_pix'] = l_pix

        # fft loss
        if self.cri_fft:
            f_pix = self.cri_fft(self.output, self.gt['Image'])
            l_total += f_pix
            loss_dict['f_pix'] = f_pix

        if self.cri_ssim:
            hr = rearrange(self.output, 'b t c h w -> (b t) c h w')
            gt = rearrange(self.gt['Image'], 'b t c h w -> (b t) c h w')
            # print(hr.shape, gt.shape)
            l_ssim = self.cri_ssim(hr, gt)
            l_total += l_ssim
            loss_dict['l_ssim'] = l_ssim

        # consisten
        if self.cri_consisten:
            l_consisten = self.cri_consisten(self.output, self.gt['Image'])
            l_total += l_consisten
            loss_dict['l_consisten'] = l_consisten

        if self.cri_time:
            l_consisten = self.cri_consisten(self.output, self.lq['Motion'])
            l_total += l_consisten
            loss_dict['l_consisten'] = l_consisten

        # perceptual loss
        if self.cri_perceptual:
            for (sr, hr) in zip(self.output.unbind(dim=1), self.gt['Image'].unbind(dim=1)):
                l_percep, l_style = self.cri_perceptual(sr, hr)

                if l_percep is not None:
                    l_total += l_percep
                    loss_dict['l_percep'] = l_percep

                if l_style is not None:
                    l_total += l_style
                    loss_dict['l_style'] = l_style

        l_total.backward()
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['result'] = self.output.detach().cpu()
        if type(self.gt['Image']) is not None:
            out_dict['gt'] = self.gt['Image'].detach().cpu()
        return out_dict


    def data_unsqueeze(self, data):
        for image_type in data:
            if 'lq' in image_type or 'gt' in image_type:
                data[image_type].unsqueeze_(0)
        return data

    def data_squeeze(self, data):
        for image_type in data:
            if 'lq' in image_type or 'gt' in image_type:
                data[image_type].squeeze_(0)
        return data

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img):
        dataset = dataloader.dataset
        dataset_name = dataset.opt['name']
        with_metrics = self.opt['val']['metrics'] is not None

        if with_metrics:
            if not hasattr(self, 'metric_results'):  # only execute in the first run
                self.metric_results = {}
                num_frame_each_folder = Counter(dataset.data_info['folder'])
                for folder, num_frame in num_frame_each_folder.items():
                    self.metric_results[folder] = torch.zeros(
                        num_frame, len(self.opt['val']['metrics']), dtype=torch.float32, device='cuda')
            # initialize the best metric results
            self._initialize_best_metric_results(dataset_name)
        # zero self.metric_results
        rank, world_size = get_dist_info()
        if with_metrics:
            for _, tensor in self.metric_results.items():
                tensor.zero_()

        metric_data = dict()
        num_folders = len(dataset)
        num_pad = (world_size - (num_folders % world_size)) % world_size
        if rank == 0:
            pbar = tqdm(total=len(dataset), unit='folder')
        # Will evaluate (num_folders + num_pad) times, but only the first num_folders results will be recorded.
        # (To avoid wait-dead)

        for i in range(rank, num_folders + num_pad, world_size):
            idx = min(i, num_folders - 1)
            val_data = dataset[idx]
            folder = val_data['folder']

            # compute outputs
            val_data = self.data_unsqueeze(val_data)
            self.feed_data(val_data)
            val_data = self.data_squeeze(val_data)

            self.test()
            visuals = self.get_current_visuals()

            # tentative for out of GPU memory
            del self.lq
            del self.output
            del self.gt

            torch.cuda.empty_cache()
            # evaluate
            if i < num_folders:
                for idx in range(visuals['result'].size(1)):
                    result_tensor = visuals['result'][0, idx, :, :, :]
                    result_img = tensor2img(result_tensor)
                    metric_data['img'] = result_img

                    gt_tensor = visuals['gt'][0, idx, :, :, :]
                    gt_img = tensor2img(gt_tensor)  #

                    metric_data['img2'] = gt_img

                    if save_img and not self.opt['is_train']:
                      img_path = osp.join(self.opt['path']['visualization'], dataset_name, folder, f"{idx:04d}_{self.opt['name']}.png")
                      imwrite(result_img, img_path)

                    # calculate metrics
                    if with_metrics:
                        for metric_idx, opt_ in enumerate(self.opt['val']['metrics'].values()):
                            result = calculate_metric(metric_data, opt_)
                            self.metric_results[folder][idx, metric_idx] += result
                # progress bar
                if rank == 0:
                    for _ in range(world_size):
                        pbar.update(1)
                        pbar.set_description(f'Folder: {folder}')

        if rank == 0:
            pbar.close()

        if with_metrics:
            if self.opt['dist']:
                # collect data among GPUs
                for _, tensor in self.metric_results.items():
                    dist.reduce(tensor, 0)
                dist.barrier()

            if rank == 0:
                self._log_validation_metric_values(current_iter, dataset_name, tb_logger)

    def segment_frame(self):
          frame_num = self.opt['val']['segment_frame']
          segmented_frames = {}

          # 将每种类型的视频帧进行切分
          for key, seq in self.lq.items():
              segmented_frames[key] = torch.split(seq, frame_num, dim=1) if seq is not None else None

          # 把分段数设为最小分段数，确保长度一致
          min_segments = min([len(segments) for segments in segmented_frames.values() if segments is not None])

          # 创建新的字典列表，每个字典包含所有类型的相应的帧序列
          net_inputs = []
          for i in range(min_segments):
              net_input = {}
              for key, segments in segmented_frames.items():
                  if segments is not None:
                      net_input[key] = segments[i]  # 选择当前分段的子序列帧
              net_inputs.append(net_input)

          return torch.cat([self.net_g(net_input).detach().cpu() for net_input in net_inputs], dim = 1)

    def test(self):
        self.net_g.eval()
        with torch.no_grad():
            if self.opt['val'].get('segment_frame'):
              self.output = self.segment_frame()
            else:
              self.output = self.net_g(self.lq)

        self.net_g.train()
