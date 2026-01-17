import math
import os

import imageio.v3
import numpy
import numpy as np
import torch
from math import exp
import torch.nn.functional as F
import subprocess

import torchvision
from torch.backends import cudnn
# This script is adapted from the following repository: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py


def calculate_psnr(img1, img2):
    """Calculate PSNR (Peak Signal-to-Noise Ratio).

    Ref: https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio

    Args:
        img1 (tensor): Images with range [0, 1.].
        img2 (tensor): Images with range [0, 1.].
        test_y_channel (bool): Test on Y channel of YCbCr. Default: False.

    Returns:
        float: psnr results.
    """
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2)) # PSNR = 10 * log10((MAX^2) / MSE)

def gaussian(window_size, sigma) :
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel) :
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    #window = torch.autograd.Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def _ssim_map(img1, img2, window, window_size, channel):
    """Calculate SSIM (structural similarity) for one channel images.

    It is called by func:`calculate_ssim`.

    SSIM(x, y) = (2 * μx * μy + c1) * (2 * σxy + c2) / ((μx^2 + μy^2 + c1) * (σx^2 + σy^2 + c2))
    其中，x和y分别表示待比较的两幅图像，μ表示图像的均值，σ^2表示方差，σxy表示协方差，c1和c2是两个常数，用于避免分母为零。

    Args:
        img1 (tensor): Images with range [0, 1.] with order 'CHW'.
        img2 (tensor): Images with range [0, 1.] with order 'CHW'.

    Returns:
        float: ssim results.
    """
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map


def calculate_ssim(img1, img2, window_size = 11, size_average = True):
    """Calculate SSIM (structural similarity).

    Args:
        img1 (tensor): Images with range [0, 1.].
        img2 (tensor): Images with range [0, 1.].

    Returns:
        float: ssim results.
    """

    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    ssim_map = _ssim_map(img1, img2, window, window_size, channel)

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def calculate_ENL(output, window_size=3, channel=1):
    window = create_window(window_size, channel)
    if output.is_cuda:
        window = window.cuda(output.get_device())
    window = window.type_as(output)
    mu = F.conv2d(output, window, padding=window_size // 2, groups=channel)
    mu_sq = mu.pow(2)
    sigma_sq = F.conv2d(output * output, window, padding=window_size // 2, groups=channel) - mu_sq

    ENL = mu_sq / sigma_sq

    return ENL.mean()

def calculate_MoI(output, noisy, window_size=11, channel=1):
    # window = create_window(window_size, channel)
    # if output.is_cuda:
    #     window = window.cuda(output.get_device())
    # window = window.type_as(output)
    # output_mu = F.conv2d(output, window, padding=window_size // 2, groups=channel)
    # noisy_mu = F.conv2d(noisy, window, padding=window_size // 2, groups=channel)
    noisy_mu = torch.mean(noisy)
    output_mu = torch.mean(output)
    moi = noisy_mu / output_mu
    return moi.mean()

def calculate_Cx(output, window_size=11, channel=1):
    # window = create_window(window_size, channel)
    # if output.is_cuda:
    #     window = window.cuda(output.get_device())
    # window = window.type_as(output)
    # output_mu = F.conv2d(output, window, padding=window_size // 2, groups=channel)
    # output_mu_sq = output_mu.pow(2)
    # output_sigma_sq = F.conv2d(output * output, window, padding=window_size // 2, groups=channel) - output_mu_sq
    # output_sigma = torch.sqrt(output_sigma_sq)
    # Cx = output_sigma / output_mu
    mean = torch.mean(output)
    std = torch.std(output)
    Cx = std / mean
    return Cx.mean()

def calculate_EPD_ROA(img, denoised_img):
    # 计算原始图像和去噪图像的HD水平方向和垂直VD方向的EPD-ROA
    I1, I2 = 0, 0
    I3, I4 = 0, 0
    for j in range(img.shape[1]):
        for i in range(img.shape[0]-1):
            if denoised_img[i+1][j] !=0:
                temp1 = np.abs(denoised_img[i][j] / denoised_img[i+1][j])
                if temp1 <= 8:
                    I1 = temp1 + I1
            if img[i + 1][j]!=0:
                temp2 = np.abs(img[i][j] / img[i + 1][j])
                if temp2 <= 8:
                    I2 = temp2 +I2
    VD = I1 / I2
    for x in range(img.shape[0]):
        for y in range(img.shape[1]-1):
            if denoised_img[x][y+1]!=0:
                temp3 = np.abs(denoised_img[x][y] / denoised_img[x][y+1])
                if temp3 <= 8:
                    I3 = temp3 + I3
            if img[x][y+1]!=0:
                temp4 = np.abs(img[x][y] / img[x][y+1])
                if temp4 <= 8:
                    I4 = temp4+I4
    HD = I3 / I4
    return HD, VD

def compute_MOR(despeckled,noisy):
    sum = 0
    for i in range(despeckled.shape[0]):
        for j in range(despeckled.shape[1]):
            if despeckled[i][j] != 0:
                sum += noisy[i][j] / despeckled[i][j]
    return sum / (despeckled.shape[0]*despeckled.shape[1])

def compute_enl(img):
    """
       计算遥感图像的ENL指标
       :param image: 遥感图像，类型为tensor
       :return: ENL指标值
    """
    # 计算像元方差
    mean = torch.mean(img)
    variance = torch.var(img)

    # 计算等效观测次数
    ENL = mean ** 2 / variance

    return ENL

if __name__ == '__main__':
    to_tensor = torchvision.transforms.ToTensor()
    output = imageio.v3.imread(r'C:\Users\admin\Desktop\speckle\port_4L_29999.tif')
    print(output.shape)
    clean = imageio.v3.imread(fr'E:\SARDiffusion\scratch\data\synthesis_v2\val\gt\20220813_port_444.tif')
    clean = (clean * 255).astype(numpy.uint8)
    ppb = imageio.v3.imread(r'C:\Users\admin\Desktop\4L-output\20220813_port_444_FANS_L4.tif')
    ppb = ppb.reshape(256,256,1)
    output = (output * 0.2 + ppb * 0.8).astype(np.uint8)
    imageio.v3.imwrite(r'C:\Users\admin\Desktop\speckle\20220813_port_444_syn.tif',output)
    # print(output.shape)
    std = np.std(output)
    ssim = calculate_ssim(to_tensor(output).unsqueeze(0), to_tensor(clean).unsqueeze(0))
    psnr = calculate_psnr(to_tensor(output).unsqueeze(0), to_tensor(clean).unsqueeze(0))
    print(ssim)
    print(psnr)
    print(std)
    # noisy = imageio.v3.imread(fr'E:\SARDiffusion\scratch\data\synthesis_v2\val\input_4L\20220813_port_444.tif')
    # noisy = (noisy * 255).astype(numpy.uint8)
    #clean = imageio.v3.imread(fr'E:\SARDiffusion\scratch\data\synthesis_v2\val\gt\20220813_port_444.tif')
    #clean = imageio.v3.imread(fr'E:\SARDiffusion\scratch\data\synthesis_v2\val\gt\20220630_farm_75.tif')
    #clean = imageio.v3.imread(fr'E:\SARDiffusion\scratch\data\synthesis_v2\val\gt\20220630_forest_83.tif')
    # clean = imageio.v3.imread(fr'E:\SARDiffusion\scratch\data\synthesis_v2\val\gt\20220813_city_32.tif')
    # print(np.min(clean))
    # print(math.log(0.039))
    # clean = (clean * 255).astype(numpy.uint8)
    #imageio.imwrite(r'C:\Users\admin\Desktop\4L-show\noisy.tif', noisy)
    #imageio.imwrite(r'C:\Users\admin\Desktop\4L-show\reference.tif', clean)