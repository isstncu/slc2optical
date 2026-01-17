import os
from os import listdir
from os.path import isfile

import imageio
import torch
import numpy as np
import torchvision
import torch.utils.data
import re
import random


class Synthsis:
    def __init__(self, config):
        self.config = config
        self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


    def get_loaders(self, parse_patches=True, validation='synthsis'):
        print("=> training SAR set...")
        train_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'synthsis', 'train'),
                                        n=self.config.training.patch_n,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=None,
                                        parse_patches=parse_patches)
        val_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'synthsis', 'val'),
                                      n=self.config.training.patch_n, # 16
                                      patch_size=self.config.data.image_size,# 64
                                      transforms=self.transforms,
                                      filelist='synthsis_val.txt',
                                      parse_patches=parse_patches)

        if not parse_patches:
            self.config.training.batch_size = 1
            self.config.sampling.batch_size = 1

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config.training.batch_size,
                                                   shuffle=True, num_workers=self.config.data.num_workers,
                                                   pin_memory=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config.sampling.batch_size,
                                                 shuffle=False, num_workers=self.config.data.num_workers,
                                                 pin_memory=True)

        return train_loader, val_loader


class SARDataset(torch.utils.data.Dataset):
    def __init__(self, dir, patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()

        if filelist is None:
            sar_dir = dir
            input_names, gt_names = [], []

            # SAR train filelist
            sar_inputs = os.path.join(sar_dir, 'input')
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            assert len(images) == 4200
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(sar_dir, 'gt'), i) for i in images]
            print(len(input_names))

            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
            self.dir = None
        else:
            self.dir = dir
            train_list = os.path.join(dir, filelist)
            with open(train_list) as f:
                contents = f.readlines()
                input_names = [i.strip() for i in contents]
                gt_names = [i.strip().replace('input', 'gt') for i in input_names]

        self.input_names = input_names
        self.gt_names = gt_names
        self.patch_size = patch_size # 64
        self.transforms = transforms # ToTensor
        self.n = n # 16
        self.parse_patches = parse_patches # True

    @staticmethod
    def get_params(img, output_size):
        w, h = img.shape  # 640x480 将img.size->img.shape，之前是PIL格式
        x, y = output_size  # 64x64
        return x, y, h, w

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(int(h / x) - 6):  # 10-6
            for j in range(int(w / y) - 6):  # 10-6
                new_crop = img[64 + i * x:64 + (i + 1) * x, 64 + j * y:64 + (j + 1) * y]
                crops.append(new_crop)
        return tuple(crops)

    def get_images(self, index):
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        img_id = re.split('/', input_name)[-1][:-4]
        # 将输入读取改为sar图像tif格式，使用imageio
        input_img = imageio.v3.imread(os.path.join(self.dir, input_name)) if self.dir else imageio.v3.imread(input_name)
        gt_img = imageio.v3.imread(os.path.join(self.dir, gt_name)) if self.dir else imageio.v3.imread(gt_name)
        # 原代码
        # input_img = PIL.Image.open(os.path.join(self.dir, input_name)) if self.dir else PIL.Image.open(input_name)
        # try:
        #     gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)) if self.dir else PIL.Image.open(gt_name)
        # except:
        #     gt_img = PIL.Image.open(os.path.join(self.dir, gt_name)).convert('RGB') if self.dir else \
        #         PIL.Image.open(gt_name).convert('RGB')
        # print(self.parse_patches)

        if self.parse_patches:
            x, y, h, w = self.get_params(input_img, (self.patch_size, self.patch_size))
            input_img = self.n_random_crops(input_img, x, y, h, w) # 16 64x64
            gt_img = self.n_random_crops(gt_img, x, y, h, w) # 16 64x64
            outputs = [torch.cat([self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                       for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            wd_new, ht_new = input_img.size
            if ht_new > wd_new and ht_new > 1024:
                wd_new = int(np.ceil(wd_new * 1024 / ht_new))
                ht_new = 1024
            elif ht_new <= wd_new and wd_new > 1024:
                ht_new = int(np.ceil(ht_new * 1024 / wd_new))
                wd_new = 1024
            wd_new = int(16 * np.ceil(wd_new / 16.0))
            ht_new = int(16 * np.ceil(ht_new / 16.0))
            input_img.resize((wd_new, ht_new))
            gt_img.resize((wd_new, ht_new))

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    # def __len__(self):
    #     return len(self.input_names)
