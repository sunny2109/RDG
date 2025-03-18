import torch
from torch import nn as nn
from torch.nn import functional as F
from einops import rearrange
from basicsr.utils.render_data_util import motion_warp

import math
from torch.nn import init
from torch.nn.modules.utils import _pair
from torchvision.ops.deform_conv import deform_conv2d as deform_conv2d_tv


class DFM(nn.Module):
    """modified from https://github.com/ShoufaChen/CycleMLP"""
    def __init__(self, dim, ratio=2):
        super(DFM, self).__init__()
        hidden_dim = int(dim * ratio)

        assert hidden_dim % 2 == 0

        self.proj_in = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 3, 1, 1, groups=dim),
            nn.Conv2d(hidden_dim, hidden_dim, 1, 1, 0),
        )

        self.mlp_h = CycleFC(dim, dim, (1, 7), 1, 0)
        self.mlp_w = CycleFC(dim, dim, (7, 1), 1, 0)

        self.merge = nn.Conv2d(hidden_dim//2 + dim, hidden_dim//2, 1, 1, 0)

        self.proj_out = nn.Conv2d(hidden_dim//2, dim, 1, 1, 0)

        self.act = nn.GELU()

    def forward(self, x):
        g, c = self.proj_in(x).chunk(2, dim=1)

        c = self.mlp_w(self.mlp_h(c))

        out = self.act(g) * self.merge(torch.cat([x, c], dim=1))

        return self.proj_out(out)

class HFB(nn.Module):
    def __init__(self, dim = 16, g_dim = 6):
        super(HFB, self).__init__()
        self.conv_g = nn.Conv2d(g_dim, dim, 3, 1, 1)
        self.conv_x = nn.Conv2d(dim, dim, 3, 1, 1)
        self.gelu = nn.GELU()
        self.proj_out = nn.Conv2d(2 * dim, dim, 1, 1, 0)

    def forward(self, x, g):
        h, w = x.shape[-2:]
        g = F.interpolate(g, size = (h, w), mode='bilinear')

        x1 = self.conv_x(x)
        x2 = self.conv_g(g)

        x = torch.cat([x1 * self.gelu(x2), x2 * self.gelu(x1)], dim = 1)

        return self.proj_out(x)


class CTR(nn.Module):
    """Temporal Attention with motion & depth guidance"""
    def __init__(self, dim=16, ratio=2.0, num_head=4):
        super(CTR, self).__init__()

        hidden = int(ratio * dim)

        self.head = num_head

        self.to_q = nn.Sequential(
            nn.Conv2d(dim, hidden, 1, 1, 0),
            nn.Conv2d(hidden, hidden, 3, 1, 1, groups=hidden)
        )

        self.to_kv = nn.Sequential(
            nn.Conv2d(dim, hidden*2, 1, 1, 0),
            nn.Conv2d(hidden*2, hidden*2, 3, 1, 1, groups=hidden*2)
        )

        self.proj_out = nn.Conv2d(hidden + 1, dim, 1, 1, 0)

        self.alpha = nn.Parameter(torch.zeros(1, 1, hidden // (num_head*2), hidden // num_head), requires_grad=True)

        self.sigm = nn.Sigmoid()

    def forward(self, x, depth, pre_h=None, flow=None):
        b, _, h, w = x.shape
        depth = F.interpolate(depth, size=(h, w), mode='bilinear')

        if pre_h is None:
          pre_h = x
          flow = x.new_zeros(b, 2, h, w)
        else:
          flow = F.interpolate(flow, size=(h, w), mode='bilinear')

        q1, q2 = self.to_q(x).chunk(2, dim=1)  #[b c h w]

        k, v = self.to_kv(motion_warp(pre_h, flow)).chunk(2, dim = 1)

        q1 = rearrange(q1, 'b (d c) h w -> b d c (h w)', d=self.head)
        k = rearrange(k, 'b (d c) h w -> b d (h w) c', d=self.head)
        v = rearrange(v, 'b (d c) h w -> b d c (h w)', d=self.head)

        q1 = F.normalize(q1, dim=-1)
        k  = F.normalize(k, dim=-2)

        out = F.softmax((q1 @ k) + 1e-7, dim = -1) @ v
        out = rearrange(out, 'b d c (h w) -> b (d c) h w', d=self.head, h=h, w=w)

        return self.proj_out(torch.cat([out, q2, depth], dim = 1))

class CycleFC(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size,  # re-defined kernel_size, represent the spatial area of staircase FC
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
    ):
        super(CycleFC, self).__init__()

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        if stride != 1:
            raise ValueError('stride must be 1')
        if padding != 0:
            raise ValueError('padding must be 0')

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups

        self.weight = nn.Parameter(torch.empty(out_channels, in_channels // groups, 1, 1))  # kernel size == 1

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.register_buffer('offset', self.gen_offset())

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))

        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def gen_offset(self):
        """
        offset (Tensor[batch_size, 2 * offset_groups * kernel_height * kernel_width,
            out_height, out_width]): offsets to be applied for each position in the
            convolution kernel.
        """
        offset = torch.empty(1, self.in_channels*2, 1, 1)
        start_idx = (self.kernel_size[0] * self.kernel_size[1]) // 2
        assert self.kernel_size[0] == 1 or self.kernel_size[1] == 1, self.kernel_size
        for i in range(self.in_channels):
            if self.kernel_size[0] == 1:
                offset[0, 2 * i + 0, 0, 0] = 0
                offset[0, 2 * i + 1, 0, 0] = (i + start_idx) % self.kernel_size[1] - (self.kernel_size[1] // 2)
            else:
                offset[0, 2 * i + 0, 0, 0] = (i + start_idx) % self.kernel_size[0] - (self.kernel_size[0] // 2)
                offset[0, 2 * i + 1, 0, 0] = 0
        return offset

    def forward(self, input):
        """
        Args:
            input (Tensor[batch_size, in_channels, in_height, in_width]): input tensor
        """
        B, C, H, W = input.size()
        return deform_conv2d_tv(input, self.offset.expand(B, -1, H, W), self.weight, self.bias, stride=self.stride,
                                padding=self.padding, dilation=self.dilation)

    def extra_repr(self) -> str:
        s = self.__class__.__name__ + '('
        s += '{in_channels}'
        s += ', {out_channels}'
        s += ', kernel_size={kernel_size}'
        s += ', stride={stride}'
        s += ', padding={padding}' if self.padding != (0, 0) else ''
        s += ', dilation={dilation}' if self.dilation != (1, 1) else ''
        s += ', groups={groups}' if self.groups != 1 else ''
        s += ', bias=False' if self.bias is None else ''
        s += ')'
        return s.format(**self.__dict__)




class CCM(nn.Module):
    def __init__(self, dim=16, ratio=2.0):
        super(CCM, self).__init__()
        hidden = int(dim * ratio)
        self.fn = nn.Sequential(
            nn.Conv2d(dim, hidden, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(hidden, dim, 1, 1, 0),
        )
    def forward(self, x):
        return self.fn(x)
