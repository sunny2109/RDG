import torch
from torch import nn as nn
from torch.nn import functional as F
from basicsr.utils.registry import ARCH_REGISTRY
from basicsr.archs.rdg_block import DFM, HFB, CTR, CCM


class EncodeLayer(nn.Module):
    def __init__(self, dim = 16):
      super(EncodeLayer, self).__init__()
      self.dfm = DFM(dim)
      self.ffn = CCM(dim)

    def forward(self, x):
        x = self.dfm(x) + x
        x = self.ffn(x) + x
        return x

class DecodeLayer(nn.Module):
    def __init__(self, dim = 16, g_dim = 6):
      super(DecodeLayer, self).__init__()

      self.hfb = HFB(dim, g_dim)
      self.ctr = CTR(dim)
      self.ffn = CCM(dim)

    def forward(self, x, g, d, pre_h=None, flow=None):
        x = self.hfb(x, g) + x
        x = self.ctr(x, d, pre_h, flow) + x
        x = self.ffn(x) + x
        return x

class Upsampling(nn.Module):
    def __init__(self, dim = 16, out_dim = 16):
      super(Upsampling, self).__init__()
      self.conv =nn.Conv2d(dim, out_dim, 3, 1, 1)
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='bilinear')
        x = self.conv(x)
        return x

@ARCH_REGISTRY.register()
class RDG(nn.Module):

    def __init__(self, in_channle=3, num_feat=16, scale = 4, middle_blk_num = 1, enc_blk_nums=[1, 1], dec_blk_nums=[1, 1]):
        super().__init__()

        self.in_channle = in_channle
        self.dim = num_feat
        self.scale = scale

        self.first_conv = nn.Conv2d(self.in_channle, self.dim, 3, 1, 1)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_encoders = nn.ModuleList()
        self.middle_decoders = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        self.skip_alphas = nn.ParameterList([nn.Parameter(torch.zeros((1, i, 1, 1))) for i in [num_feat, num_feat*2]])

        chan = self.dim
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[EncodeLayer(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, chan + self.dim, 3, 2, 1)
            )
            chan += self.dim

        self.middle_encoders = nn.Sequential(
                *[EncodeLayer(chan) for _ in range(middle_blk_num)]
            )

        self.middle_decoders = DecodeLayer(dim = chan, g_dim = 6)

        for num in dec_blk_nums:
            self.ups.append(
              Upsampling(chan, chan - self.dim)
            )
            chan -= self.dim
            self.decoders.append(
              DecodeLayer(dim = chan, g_dim = 6)
            )
        self.padder_size = 2 ** len(self.encoders)

        self.UpSample = nn.Sequential(
            nn.Conv2d(self.dim,  3 * self.scale * self.scale, 3, 1, 1),
            nn.PixelShuffle(self.scale)
        )

    def forward(self, input_data):
        images = input_data['Image']  # (b, t, c, h, w)
        motions = input_data['Motion']
        depths = input_data['Depth']
        gbuffers = torch.cat([input_data['Normal'], input_data['BRDF']], dim = 2)

        out_l = []
        pre_hs = [None, None, None]
        for i in range(0, images.shape[1]):
            x_cur = images[:, i, ...]
            g_cur = gbuffers[:, i, ...]
            m_cur = motions[:, i - 1, ...] if i > 0 else None
            d_cur = depths[:, i, ...]

            out, pre_hs= self.process_forward(x_cur, g_cur, m_cur, d_cur, pre_hs)
            out_l.append(out)

        return torch.stack(out_l, dim=1)


    def process_forward(self, x, g, m, d, pre_hs):
        H, W = x.shape[-2:]
        x, g, m, d = map(lambda t: self.check_image_size(t), (x, g, m, d))

        x = self.first_conv(x)

        res = x
        encs = []
        cur_hs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        x = self.middle_encoders(x)
        x = self.middle_decoders(x, g, d, pre_hs[0], m)
        cur_hs.append(x)

        for decoder, up, alpha, enc_skip, pre_h in zip(self.decoders, self.ups, self.skip_alphas[::-1], encs[::-1], pre_hs[1:]):
            x = up(x)
            x = x + alpha * enc_skip
            x = decoder(x, g, d, pre_h, m)
            cur_hs.append(x)

        x = x + res
        return self.UpSample(x[:, :, :H, :W]), cur_hs

    def check_image_size(self, x):
        if x is None:
            return None
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x




if __name__== '__main__':
    #############Test Model Complexity #############
    # import time
    from fvcore.nn import flop_count_table, FlopCountAnalysis, ActivationCountAnalysis
    from tqdm import tqdm
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    torch.cuda.current_device()
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False

    scale = 4
    clip = 200
    h, w =  1920, 1080
    num_frame = 5
    dummy_input =  {
        'Image':  torch.randn((1, num_frame, 3, 1920 // scale, 1080 // scale)).cuda(),
        'Motion':  torch.randn((1, num_frame, 2, 1920 // scale, 1080 // scale)).cuda() ,
        'Depth':  torch.randn((1, num_frame, 1, 1920 // scale, 1080 // scale)).cuda(),
        'Normal':  torch.randn((1, num_frame, 3, 1920 // scale, 1080 // scale)).cuda() ,
        'BRDF':  torch.randn((1, num_frame, 3, 1920 // scale, 1080 // scale)).cuda() ,
        'gt_Normal':  torch.randn((1, num_frame, 3, 1920, 1080)) ,
        'gt_BRDF':  torch.randn((1, num_frame, 3, 1920 , 1080 )) ,
        'gt_Depth':  torch.randn((1, num_frame, 1, 1920 , 1080 )) ,
    }

    model = RDG(num_feat=16).cuda()
    model.eval()



    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    runtime = 0

    #  model.eval()
    with torch.no_grad():
      # print(model)
      for _ in tqdm(range(clip)):
          _ = model(dummy_input)

      for _ in tqdm(range(clip)):
          start.record()
          _ = model(dummy_input)
          end.record()
          torch.cuda.synchronize()
          runtime += start.elapsed_time(end)

      per_frame_time = runtime / (num_frame * clip)
      max_memory = torch.cuda.max_memory_allocated(torch.cuda.current_device()) / 1024 ** 2

      print(f'{num_frame * clip} Number Frames x{scale}SR Per Frame Time: {per_frame_time:.6f} ms')
      print(f' x{scale}SR FPS: {(1000 / per_frame_time):.6f} FPS')
      print(f' Max Memery {max_memory / num_frame :.6f} [M]')
      output = model(dummy_input)
      print(output.shape)
      print(flop_count_table(FlopCountAnalysis(model, dummy_input), activations=ActivationCountAnalysis(model, dummy_input)))









