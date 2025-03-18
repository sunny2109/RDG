import cv2
import numpy as np
import torch
import random
from torch.nn import functional as F
from basicsr.utils.color_util import rgb2ycbcr

def read_exr_bgr2rgb(path):
    """读取EXR格式图像并转换为RGB格式。
    参数:
        path: EXR图像的路径。
    返回:
        图像的RGB表示。
    """
    if path is None:
        raise ValueError('Image path is None.')

    bgr = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if bgr is None:
        raise ValueError(f'Image from {path} is None.')

    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb

def load_image(path, format='rgb'):
    """

    参数:
        path (str): 图像的路径。
        format (str): 返回图像的格式 ('rgb' 或 'ycbcr')。


    返回:
        numpy.ndarray: 根据指定格式返回的图像数据。
    """
    rgb = read_exr_bgr2rgb(path)[:,:,:3] / 255.  # 预先执行归一化步骤

    if format == 'rgb':
        return rgb
    elif format == 'ycbcr':
        return rgb2ycbcr(rgb)  # 预先转换图像，然后返回
    else:
        raise ValueError(f'Unsupported image format: {format}')


def load_exr(path, buffer_type):
    """根据指定的缓冲类型加载并处理EXR格式的图像。
    参数:
        path: EXR图像的路径。
        buffer_type: 指的是图像的类型，如 'Normal', 'Depth' 等。
    返回:
        处理后的图像数据。
    """
    image = read_exr_bgr2rgb(path)

    # if buffer_type in ['Normal', 'Emit', 'BRDF']:
    #     return image[:,:,:3]
    # elif buffer_type == 'Motion':
    #     image[:,:,1] *= -1  # 修正Y方向
    #     return image[:,:,:2]
    # elif buffer_type == 'Depth':
    #     x = (image[:,:,0][:,:,None] - np.min(image)) / (np.max(image) - np.min(image))
    #     return x
    # else:
    #     raise ValueError(f'{buffer_type} is an unknown buffer type!')


    if buffer_type in ['Emit', 'BRDF']:
      return (image - np.min(image)) / (np.max(image) - np.min(image))

    elif buffer_type == 'Normal':
      return (image[:,:,:3] + 1)  / 2

    elif buffer_type == 'Motion':
      image[:,:,1] *= -1  # 修正Y方向
      return image[:,:,:2]

    elif buffer_type == 'Depth':
      return (image[:,:,0][:,:,None] - np.min(image)) / (np.max(image) - np.min(image))

    else:
        raise ValueError(f'{buffer_type} is an unknown buffer type!')




def imgs2tensor(imgs):
  return torch.stack([torch.from_numpy(img).permute(2, 0, 1).float() for img in imgs], dim=0)

def load_img_seq(path_list, image_type = 'Image', format = 'rgb'):
    if image_type == 'Image':
      imgs = [load_image(path, format) for path in path_list]
      return imgs2tensor(imgs)
    else:
      imgs = [load_exr(path, image_type) for path in path_list]
      return imgs2tensor(imgs)

def np2tensor(lqs, gts):
      for image_type in lqs:
          lqs[image_type] = imgs2tensor(lqs[image_type])
      for image_type in gts:
          gts[image_type] = imgs2tensor(gts[image_type])
      return lqs, gts

def paired_random_crop(lqs, gts, gt_patch_size, scale):
    h_lq, w_lq = lqs['Image'][0].shape[0:2]
    h_gt, w_gt = gts['Image'][0].shape[0:2]

    lq_patch_size = gt_patch_size // scale

    if h_gt != h_lq * scale or w_gt != w_lq * scale:
        raise ValueError(f'Scale mismatches. GT ({h_gt}, {w_gt}) is not {scale}x ',
                        f'multiplication of LQ ({h_lq}, {w_lq}).')
    if h_lq < lq_patch_size or w_lq < lq_patch_size:
      raise ValueError(f'LQ ({h_lq}, {w_lq}) is smaller than patch size '
                      f'({lq_patch_size}, {lq_patch_size}). ')

    top = random.randint(0, h_lq - lq_patch_size)
    left = random.randint(0, w_lq - lq_patch_size)

    for type, image_list in lqs.items():
      lqs[type] =  [v[top:top + lq_patch_size, left:left + lq_patch_size, ...] for v in image_list]

    top_gt, left_gt = int(top * scale), int(left * scale)
    for type, image_list in gts.items():
      gts[type] =  [v[top_gt:top_gt + gt_patch_size, left_gt:left_gt + gt_patch_size, ...] for v in image_list]
    return lqs, gts


def demodulate(image, brdf):
    """
    根据image和brdf计算irradiation图像。
    这里image和brdf已经Tensor。
    """
    if brdf is None:
        raise ValueError('BRDF is None.')
    assert image.shape == brdf.shape

    brdf = F.softplus(brdf, beta=100)
    irradiance = torch.where(brdf == 0, brdf, image / brdf)
    return irradiance


def pack(lqs, gts, scence = None, use_irradiance = False):
    data = {}
    for image_type in lqs:
      data['lq_' + image_type] = lqs[image_type]
    for image_type in gts:
      data['gt_' + image_type] = gts[image_type]
    if scence is not None:
       data['folder'] = scence

    return data



def motion_warp(x, motion, interp_mode='bilinear', padding_mode='zeros', align_corners=True):
    _, _, h, w = x.size()
    # x = [b,  c, h, w], motion = [b, 2, h, w]
    if x.size()[-2:] != motion.size()[-2:]:
      motion = F.interpolate(motion, size=(h, w), mode='bilinear')

    motion = motion.permute(0, 2, 3, 1) # motion = [b, h, w, 2]

    # create mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h).type_as(x), torch.arange(0, w).type_as(x))
    grid = torch.stack((grid_x, grid_y), 2).float()  # W(x), H(y), 2
    grid.requires_grad = False

    vgrid = grid + motion
    # scale grid to [-1,1]
    vgrid_x = 2.0 * vgrid[:, :, :, 0] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * vgrid[:, :, :, 1] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode=interp_mode, padding_mode=padding_mode, align_corners=align_corners)

    return output
