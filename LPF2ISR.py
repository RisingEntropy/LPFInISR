import torch
import math
import torch.nn.functional as F

def sinc(tensor, omega):
    """
    The sinc function implementation. sinc(t) is defined as sin(pi*t)/(pi*t), omega is a
    factor to adjust the scale
    :param tensor: variants of sinc function
    :param omega: scale factor
    :return:
    """
    return torch.sin(torch.abs(math.pi * tensor * omega) + 1e-9) / (torch.abs(math.pi * tensor * omega) + 1e-9)


def nearest_odd(num):
    return num + 1 if num % 2 == 0 else num


def zero_interpolate_torch(img: torch.Tensor, scale: int):
    """
    interpolate 0 by `scale` times
    :param img: NxCxHxW
    :param scale:
    :return:
    """
    if len(img.shape) != 4:  # batched
        img = img.unsqueeze(dim=0)
    img_ = img.reshape(-1, 1, img.shape[2], img.shape[3])
    img_int = torch.concat(
        [img_, torch.zeros(img_.shape[0], scale * scale - 1, img_.shape[2], img_.shape[3]).to(img.device)],
        dim=1)
    return torch.nn.functional.pixel_shuffle(img_int, scale).reshape(img.shape[0], img.shape[1], img.shape[2] * scale,
                                                                     img.shape[3] * scale).squeeze(dim=0)


def lpf_sr_single(img: torch.Tensor, scale: int, omega=3.):
    """
    Interpolate an image using the sinc function, it's slower than the cubic or others.

    :param img: the image to be interpolated.
    :param size: the expected size
    :param omega: the factor to adjust the scale of the sinc function
    :return: the interpolated image
    :param backend: use torch or cuda code to apply zero-interpolate
    """
    img_pad = F.pad(input=img,
                    pad=(img.shape[2] // 2, img.shape[2] // 2, img.shape[3] // 2, img.shape[3] // 2),
                    mode="reflect")
    target = zero_interpolate_torch(img_pad, scale)
    h_grid = torch.linspace(-1, 1, (img.shape[2] // 2) * scale * 2 + 1)
    w_grid = torch.linspace(-1, 1, (img.shape[3] // 2) * scale * 2 + 1)
    kernel = torch.meshgrid([h_grid, w_grid], indexing='xy')

    kernel = sinc(kernel[0], omega) * sinc(kernel[1], omega)
    kernel = kernel.unsqueeze(dim=0).unsqueeze(dim=0).to(img.device)
    # kernel.require_grad = False
    target = F.conv2d(input=target, weight=kernel, stride=1, padding="valid")
    for i in range(target.shape[0]):
        if torch.max(img[i])>0.001:  # to avoid a all 0 image
            target[i] = (target[i] - torch.min(target[i]))/(torch.max(target[i])-torch.min(target[i])) * (torch.max(img[i])-torch.min(img[i])) + torch.min(img[i])
    return target


def lpf_sr(img: torch.Tensor, scale: int, omega=3.):
    """
        Interpolate image(s) using the sinc function, it's slower than the cubic or others.
        :param img: the image to be interpolated.
        :param size: the expected size
        :param omega: the factor to adjust the scale of the sinc function
        :return: the interpolated image
        """
    if len(img.shape) == 4:  # Batched
        origin_shape = img.shape
        img = img.view(-1, 1, img.shape[2], img.shape[3])
        out = lpf_sr_single(img, scale, omega)
        return out.reshape(origin_shape[0], origin_shape[1], origin_shape[2] * scale,
                                           origin_shape[3] * scale)
    else:
        origin_shape = img.shape
        img = img.view(-1, 1, img.shape[1], img.shape[2])
        out = lpf_sr_single(img, scale, omega)
        return out.reshape(origin_shape[0], origin_shape[1] * scale, origin_shape[2] * scale)