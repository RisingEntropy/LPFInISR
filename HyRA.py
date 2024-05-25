import os

import matplotlib
import torch
from PIL import Image
from einops import einops
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import matplotlib.pyplot as plt
from utils import zero_interpolate_torch
import argparse


def normalize(ten:torch.Tensor):
    return (ten - torch.min(ten)) / (torch.max(ten) - torch.min(ten))


def convertNp(img:torch.Tensor):
    return einops.rearrange(img, "C H W -> H W C").numpy()


def get_linear(lr:torch.Tensor, impulse_response:torch.Tensor, scale):
    impulse_response = impulse_response.unsqueeze(dim=1)
    lr_pad = F.pad(input=lr,
                   pad=(impulse_response.shape[2] // (2 * scale), impulse_response.shape[2] // (2 * scale),
                        impulse_response.shape[3] // (2 * scale), impulse_response.shape[3] // (2 * scale)),
                   mode="reflect")
    lr_inter = zero_interpolate_torch(lr_pad, scale)
    lr_lp = F.conv2d(input=lr_inter, weight=impulse_response, stride=1, padding="valid", groups=3)
    return lr_lp


def get_nonlinear(linear:torch.Tensor, sr:torch.Tensor):
    sr = F.interpolate(sr.unsqueeze(dim=0), size=(linear.shape[1], linear.shape[2]))
    return (sr - linear).squeeze()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=str, required=True)
    parser.add_argument("--impulse_response", type=str, required=True)
    parser.add_argument("--sr", type=str, required=True)
    parser.add_argument("--scale", type=int, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    if args.scale <= 0:
        raise ValueError("Scale must be positive")
    converter = ToTensor()
    
    lr = converter(Image.open(args.lr))
    impulse_response = converter(Image.open(args.impulse_response))
    sr = converter(Image.open(args.sr))

    cmp = matplotlib.colormaps["Blues"]
    cmp = cmp.reversed()
    linear = get_linear(lr, impulse_response, args.scale)
    non_linear = get_nonlinear(linear, sr)
    linear_fft = torch.log(torch.fft.fftshift(torch.fft.fft2(linear)).abs() + 1)
    non_linear_fft = torch.log(torch.fft.fftshift(torch.fft.fft2(non_linear)).abs() + 1)
    lr = F.interpolate(lr.unsqueeze(dim=0), size=(linear.shape[1], linear.shape[2]))[0]
    lr_fft = torch.log(torch.fft.fftshift(torch.fft.fft2(lr)).abs() + 1)
    sr_fft = torch.log(torch.fft.fftshift(torch.fft.fft2(sr)).abs() + 1)
    plt.imsave(os.path.join(args.save_path, "linear.png"), convertNp(linear / torch.max(linear)))
    plt.imsave(os.path.join(args.save_path, "non_linear.png"), convertNp(normalize(non_linear)))
    plt.imsave(os.path.join(args.save_path, "linear_fft.png"), linear_fft[0], cmap=cmp)  # display only one channel
    plt.imsave(os.path.join(args.save_path, "non_linear_fft.png"), non_linear_fft[0], cmap=cmp)
    plt.imsave(os.path.join(args.save_path, "lr_fft.png"), lr_fft[0], cmap=cmp)
    plt.imsave(os.path.join(args.save_path, "hr_fft.png"), sr_fft[0], cmap=cmp)
    print("done!")


if __name__ == "__main__":
    main()