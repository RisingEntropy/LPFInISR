import os.path
import warnings
from datetime import datetime

import einops
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


def lpf_sr_single(img: torch.Tensor, scale: int, omega=3., rgb_range = 255):
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
        if torch.max(img[i])>=0.01:  # to avoid a all 0 image
            target[i] = (target[i] - torch.min(target[i]))/(torch.max(target[i])-torch.min(target[i])) * (torch.max(img[i])-torch.min(img[i])) + torch.min(img[i])
    return target



def calc_psnr(sr, hr, scale, rgb_range, dataset=None):
    if hr.nelement() == 1: return 0

    diff = (sr - hr) / rgb_range
    if dataset and dataset.dataset.benchmark:
        shave = scale
        if diff.size(1) > 1:
            gray_coeffs = [65.738, 129.057, 25.064]
            convert = diff.new_tensor(gray_coeffs).view(1, 3, 1, 1) / 256
            diff = diff.mul(convert).sum(dim=1)
    else:
        shave = scale + 6

    valid = diff[..., shave:-shave, shave:-shave]
    mse = valid.pow(2).mean()
    if abs(mse)<1e-6:
        return 100
    return -10 * math.log10(mse)


def psnr(a, b, range=255):
    return calc_psnr(a, b, 2, range)


def lpf_sr(img: torch.Tensor, scale: int, omega=3., rgb_range=1):
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
        out = lpf_sr_single(img, scale, omega, rgb_range=rgb_range)
        return out.reshape(origin_shape[0], origin_shape[1], origin_shape[2] * scale,
                                           origin_shape[3] * scale)
    else:
        origin_shape = img.shape
        img = img.view(-1, 1, img.shape[1], img.shape[2])
        out = lpf_sr_single(img, scale, omega, rgb_range=rgb_range)
        return out.reshape(origin_shape[0], origin_shape[1] * scale, origin_shape[2] * scale)


def quantize(img, rgb_range):
    pixel_range = 255 / rgb_range
    return img.mul(pixel_range).clamp(0, 255).round().div(pixel_range)


class FileLogger:
    def __init__(self, exp_name):
        if not os.path.exists(os.path.join("ExperimentLogs", exp_name)):
            os.makedirs(os.path.join("ExperimentLogs", exp_name))
        self.exp_name = exp_name
        self.filename = os.path.join("ExperimentLogs", exp_name, "log output.txt")
        self.figure_file = os.path.join("ExperimentLogs", exp_name, "figure_curves.pt")
        self.figures = {}

    def print(self, text, append=True, sync_with_screen=True):
        with open(self.filename, "a" if append else "w") as f:
            print(datetime.now().strftime("%Y-%M-%d %H:%M:%S--->") + text, file=f)
        if sync_with_screen:
            print(datetime.now().strftime("%Y-%M-%d %H:%M:%S--->") + text)

    def log_figure(self, figure_name, figure: float):
        if figure_name in self.figures:
            self.figures[figure_name].append(figure)
        else:
            self.figures[figure_name] = [figure]
        torch.save(self.figures, self.figure_file)

    def save_model(self, obj, attribute: str = ""):
        torch.save(obj, os.path.join("ExperimentLogs", self.exp_name, f"check_point_{attribute}"))

def torch2np(tensor):
    return einops.rearrange(tensor, "C H W -> H W C").cpu().numpy()

def min_max_normalization(tensor):
    return (tensor-torch.min(tensor))/(torch.max(tensor)-torch.min(tensor))

def mean_normalization(tensor):
    return (tensor - torch.mean(tensor)) / (torch.max(tensor) - torch.min(tensor))