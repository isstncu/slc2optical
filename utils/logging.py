import imageio
import numpy
import numpy as np
import torch
import os

import utils


@torch.no_grad()
def inverse_data_transform(X):
    # return (np.transpose(X, (1, 2, 0)) + 1) / 2.0 * 255.0 # torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)
    # v2 v3 v4 v5 v6
    # array = (np.transpose(X, (1, 2, 0)) + 1) / 2.0 ×
    # v7
    array = np.transpose(X, (1, 2, 0))
    out = np.clip(array, 0.0, 1.0)
    return out * 10000.0

def save_image(img, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    # tvu.save_image(img, file_directory)
    # sar图像存储
    # Gamma
    print(torch.max(img), torch.min(img))
    # img = torch.clamp(img, 0.0, 1.0)
    # img = img * torch.log(torch.tensor(255.))
    # img = torch.exp(img)
    # print(torch.max(img).item(),torch.min(img).item())
    # pred = img.squeeze().permute(1, 2, 0).cpu().numpy()
    # print('save shape:', pred.shape)
    # results = np.zeros((pred.shape[0], pred.shape[1], 3))
    # results[:, :, 0] = pred[:, :, 0]
    # results[:, :, 1] = pred[:, :, 1]
    # results[:, :, 2] = pred[:, :, 0] / pred[:, :, 1]
    # print(img.shape)
    img = img.squeeze().cpu().numpy()
    img = inverse_data_transform(img).astype(numpy.uint16)
    # img = (img*255).astype(numpy.uint8)
    # img = denormalize_sar(img) # 指数变换转回SAR数据形式存储
    imageio.v3.imwrite(file_directory, img)

def save_image_v2(img, lambda_, file_directory):
    if not os.path.exists(os.path.dirname(file_directory)):
        os.makedirs(os.path.dirname(file_directory))
    img = torch.clamp(img, 0.0, 1.0)
    # print(img.shape)
    img = img.reshape(img.shape[-2], img.shape[-1])
    img = img.squeeze().cpu().numpy()
    img = utils.inverse_x0_process_v2(img,lambda_)
    imageio.v3.imwrite(file_directory, img)

def save_checkpoint(state, filename):
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename))
    torch.save(state, filename + '.pth.tar')


def load_checkpoint(path, device):
    if device is None:
        return torch.load(path)
    else:
        return torch.load(path, map_location=device)
