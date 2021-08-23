from chainercv.transforms import resize
import random
import PIL
import numpy as np


def letterbox_img(img, size, fill=0, interpolation=PIL.Image.BILINEAR,
                   return_param=False):

    C, H, W = img.shape
    out_H, out_W = size
    scale_h = out_H / H
    scale_w = out_W / W
    scale = min(scale_h, scale_w)
    scaled_size = (int(H * scale), int(W * scale))

    img = resize(img, scaled_size, interpolation)
    y_slice, x_slice = _get_pad_slice(img, size=size)
    out_img = np.empty((C, out_H, out_W), dtype=img.dtype)
    out_img[:] = np.array(fill).reshape((-1, 1, 1))
    out_img[:, y_slice, x_slice] = img

    if return_param:
        param = {'y_offset': y_slice.start, 'x_offset': x_slice.start,
                 'scaled_size': scaled_size}
        return out_img, param
    else:
        return out_img


def random_crop(
        img, min_scale=0.3, max_scale=1,
        max_aspect_ratio=2, return_param=False):
    """Crop an image randomly with bounding box constraints.

    This data augmentation is used in training of
    Single Shot Multibox Detector [#]_. More details can be found in
    data augmentation section of the original paper.

    .. [#] Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    """

    _, H, W = img.shape
    scale = random.uniform(min_scale, max_scale)
    aspect_ratio = random.uniform(
        max(1 / max_aspect_ratio, scale * scale),
        min(max_aspect_ratio, 1 / (scale * scale)))
    crop_h = int(H * scale / np.sqrt(aspect_ratio))
    crop_w = int(W * scale * np.sqrt(aspect_ratio))

    crop_t = random.randrange(H - crop_h)
    crop_l = random.randrange(W - crop_w)

    param = {'y_slice': slice(crop_t, crop_t + crop_h),
             'x_slice': slice(crop_l, crop_l + crop_w)}

    img = img[:, param['y_slice'], param['x_slice']]

    if return_param:
        return img, param
    else:
        return img


def _get_pad_slice(img, size):
    _, H, W = img.shape

    if H < size[0]:
        margin_y = (size[0] - H) // 2
    else:
        margin_y = 0
    y_slice = slice(margin_y, margin_y + H)

    if W < size[1]:
        margin_x = (size[1] - W) // 2
    else:
        margin_x = 0
    x_slice = slice(margin_x, margin_x + W)

    return y_slice, x_slice
