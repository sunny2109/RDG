import numpy as np
import random
import torch
from pathlib import Path
from torch.utils import data as data
import os.path as osp
from basicsr.utils.render_data_util import load_image, load_exr, load_img_seq, np2tensor, paired_random_crop, pack, demodulate
from basicsr.utils import  get_root_logger
from basicsr.utils.registry import DATASET_REGISTRY
import os
import cv2

@DATASET_REGISTRY.register()
class RenderRecurrentDataset(data.Dataset):
    def __init__(self, opt):
        super(RenderRecurrentDataset, self).__init__()
        os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
        self.opt = opt
        self.keys = []
        self.gt_root, self.lq_root = Path(opt['dataroot_gt']), Path(opt['dataroot_lq'])
        self.num_frame = opt['num_frame'] #训练的帧数
        self.format = 'ycbcr' if opt.get('is_ycbcr', False) else 'rgb'
        self.use_demodulate = opt.get('use_demodulate', False)
        self.keys = self.get_keys(opt['meta_info_file'])
        self.use_gt_buffer = self.get_buffer_usage(opt, 'gt')
        self.use_lq_buffer = self.get_buffer_usage(opt, 'lq')

    @staticmethod
    def get_keys(meta_info_file):
        keys = []
        with open(meta_info_file, 'r') as fin:
            for line in fin:

                scene, total_frame_num, patch_num, _ = line.split()
                keys.extend([f"{scene}/{total_frame_num}/{patch_num}/{i}" for i in range(int(total_frame_num))])
        return keys

    @staticmethod
    def get_buffer_usage(opt, prefix):
        buffer_types = ['Normal', 'BRDF', 'Depth', 'Motion', 'Emit']
        return {b: opt.get(f'use_{prefix}_{b.lower()}', False) for b in buffer_types}

    def init_buffers(self, use_buffer):
        outputs = {'Image': []}  # 初始化“Image”作为必须使用的类型
        for buffer, use in use_buffer.items():
            if use:
                outputs[buffer] = []  # 为其他标记为使用的类型添加空列表
        return outputs

    def __getitem__(self, index):
        scale = self.opt['scale']
        gt_size = self.opt['gt_size']
        clip_name, total_frame_num, patch_num, frame_name = self.keys[index].split('/') # key example: 00/100/12/0001.png

        # ensure not exceeding the borders
        start_frame_idx = int(frame_name)
        total_frame_num = int(total_frame_num)
        if start_frame_idx > total_frame_num - self.num_frame:
            start_frame_idx = random.randint(0, total_frame_num - self.num_frame)
        end_frame_idx = start_frame_idx + self.num_frame
        neighbor_list = list(range(start_frame_idx, end_frame_idx))

        # select the patch in the same position
        patch_idx = '_s{:03d}'.format(random.randrange(0, int(patch_num))) if int(patch_num) > 0 else ''

        lqs = self.init_buffers(self.use_lq_buffer)
        gts = self.init_buffers(self.use_gt_buffer)

        for i in neighbor_list:
            frame_str = f"{i:03d}"
            get_img_path = lambda root, buffer, ext: osp.join(root, clip_name, buffer, frame_str + patch_idx + ext)

            lqs['Image'].append(load_image(get_img_path(self.lq_root, 'Image', '.png'), self.format))
            gts['Image'].append(load_image(get_img_path(self.gt_root, 'Image', '.png'), self.format))

            for buffer_type in self.use_lq_buffer:
                if self.use_lq_buffer[buffer_type]:
                    lqs[buffer_type].append(load_exr(get_img_path(self.lq_root, buffer_type, '.exr'), buffer_type))

            for buffer_type in self.use_gt_buffer:
                if self.use_gt_buffer[buffer_type]:
                    gts[buffer_type].append(load_exr(get_img_path(self.gt_root, buffer_type, '.exr'), buffer_type))

        lqs, gts = paired_random_crop(lqs, gts, gt_size, scale)
        lqs, gts = np2tensor(lqs, gts)

        if self.use_demodulate:
            lqs['Irradiance'] = demodulate(lqs['Image'], lqs['BRDF'])
            gts['Irradiance'] = demodulate(gts['Image'], gts['BRDF'])

        return pack(lqs, gts)

    def __len__(self):
        return len(self.keys)


@DATASET_REGISTRY.register()
class RenderRecurrentTestDataset(data.Dataset):
    def __init__(self, opt):
        super(RenderRecurrentTestDataset, self).__init__()
        os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

        self.opt = opt
        self.folders = []
        self.data_info = {'folder': []}
        self.cache_data = opt.get('cache_data', False)
        self.gt_root, self.lq_root = opt['dataroot_gt'], opt['dataroot_lq']
        self.format = 'ycbcr' if opt.get('is_ycbcr', False) else 'rgb'
        self.use_demodulate = opt.get('use_demodulate', False)

        self.use_gt_buffer = self.get_buffer_usage(opt, 'gt')
        self.use_lq_buffer = self.get_buffer_usage(opt, 'lq')

        logger = get_root_logger()
        logger.info(f'Generate data info for VideoTestDataset - {opt["name"]}')

        image_path_list = self.parse_meta_info(opt['meta_info_file'])
        self.lqs_path, self.gts_path = self.init_path()
        self.load_image_paths(image_path_list)

        if self.cache_data:
            logger.info(f'Cache TestDataset ...')
            self.lqs_data, self.gts_data = self.load_all_data()

    @staticmethod
    def get_buffer_usage(opt, prefix):
        buffer_types = ['Normal', 'BRDF', 'Depth', 'Motion', 'Emit']
        return {b: opt.get(f'use_{prefix}_{b.lower()}', False) for b in buffer_types}

    def parse_meta_info(self, meta_info_file):
        if meta_info_file is None:
            raise ValueError('meta_info_file is None.')

        image_path_list = []
        with open(meta_info_file, 'r') as fin:
            for line in fin:
                scene, frame_num, start_index= line.strip().split(' ')
                image_path_list.extend([osp.join(scene, f"{i+int(start_index):03d}") for i in range(int(frame_num))])
                self.folders.append(scene)
        return image_path_list

    def init_path(self):
        lqs_path = {scene: {'Image': []} for scene in self.folders}
        gts_path = {scene: {'Image': []} for scene in self.folders}

        for scene in self.folders:
            for buffer_type, use in self.use_lq_buffer.items():
                if use: lqs_path[scene][buffer_type] = []
            for buffer_type, use in self.use_gt_buffer.items():
                if use: gts_path[scene][buffer_type] = []

        return lqs_path, gts_path

    def load_image_paths(self, image_path_list):
        for image_path in image_path_list:
            scene, image_name = image_path.split('/')
            self.data_info['folder'].append(scene)

            self.add_image_path(self.lq_root, scene, image_name, self.lqs_path, 'lq')
            self.add_image_path(self.gt_root, scene, image_name, self.gts_path, 'gt')

    def add_image_path(self, root_path, scene, image_name, paths_dict,  image_type_prefix):
        img_main_path = osp.join(root_path, scene, 'Image', f"{image_name}.png")
        paths_dict[scene]['Image'].append(img_main_path)

        for buffer_type, use in getattr(self, f'use_{image_type_prefix}_buffer').items():
            if use:
                buffer_path = osp.join(root_path, scene, buffer_type, f"{image_name}.exr")
                paths_dict[scene][buffer_type].append(buffer_path)

    def load_all_data(self):

        loaded_lqs = {scene: self.load_image_folder(folder) for scene, folder in self.lqs_path.items()}
        loaded_gts = {scene: self.load_image_folder(folder) for scene, folder in self.gts_path.items()}

        return loaded_lqs, loaded_gts

    def load_image_folder(self, image_folder):
        loaded_folder = {}
        for image_type, image_path_list in image_folder.items():
            loaded_folder[image_type] = load_img_seq(image_path_list, image_type, self.format)

          # 执行解调操作
        if self.use_demodulate:
            loaded_folder['Irradiance'] = demodulate(loaded_folder['Image'], loaded_folder['BRDF'])

        return loaded_folder

    def __getitem__(self, index):
        scene = self.folders[index]
        if self.cache_data:
            data_lqs, data_gts = self.lqs_data[scene], self.gts_data[scene]
        else:
            data_lqs = self.load_image_folder(self.lqs_path[scene])
            data_gts = self.load_image_folder(self.gts_path[scene])
        return pack(data_lqs, data_gts, scene)

    def __len__(self):
        return len(self.folders)