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
from torchvision import transforms

import utils


class SAR:
    def __init__(self, config):
        self.config = config
        # Get Probability
        p_h = random.randint(0, 1)
        p_v = random.randint(0, 1)

        # Data Augmentation
        if config.diffusion.mode == "train":
            self.transforms = transforms.Compose([
                transforms.ToTensor(),
                #transforms.RandomHorizontalFlip(p=p_h),
                #transforms.RandomVerticalFlip(p=p_v),
            ])

        elif config.diffusion.mode == "valid":
            self.transforms = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])


    def get_loaders(self, parse_patches=True):
        if parse_patches == True:
            print("=> training {} set...".format(self.config.training.name))
        else:
            print("=> evaluating SAR set...")
        # train_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'synthesis_v2', 'train'),
        train_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'train'),
                                        n=self.config.training.patch_n, # 16
                                        input_channel = self.config.data.channels,
                                        patch_size=self.config.data.image_size,
                                        transforms=self.transforms,
                                        filelist=None,
                                        parse_patches=parse_patches)
        val_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'eval'),
        # val_dataset = SARDataset(dir=os.path.join(self.config.data.data_dir, 'data', 'synthesis_v2', 'val'),
                                      n=self.config.training.patch_n, # 16
                                      input_channel= self.config.data.channels,
                                      patch_size=self.config.data.image_size,# 64
                                      transforms=self.transforms,
                                      filelist=self.config.sampling.filelist,
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
        # return val_loader


class SARDataset(torch.utils.data.Dataset):
    def __init__(self, dir, input_channel,patch_size, n, transforms, filelist=None, parse_patches=True):
        super().__init__()
        # input_names_VH, input_names_VV, gt_names = [], [], []
        input_names, gt_names, input_names_VH, input_names_VV, = [], [], [], []
        self.input_channel = input_channel

        if filelist is None:
            sar_dir = dir
            # SAR train
            # Complex SAR
            # sar_inputs = os.path.join(sar_dir, 'S1')
            # C2Matrix
            sar_inputs = os.path.join(sar_dir, 'C2Matrix')
            # normalized
            # sar_inputs = os.path.join(sar_dir, 'S1_normalize')
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            assert len(images) == 2183 # 1910
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(sar_dir, 'S2'), i.replace('S1', 'S2')) for i in images]
            #
            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]

            # GRD SAR
            # sar_inputs_VH = os.path.join(sar_dir, 'GRD', 'VH')
            # sar_inputs_VV = os.path.join(sar_dir, 'GRD', 'VV')
            # images_VH = [f for f in listdir(sar_inputs_VH) if isfile(os.path.join(sar_inputs_VH, f))]
            # images_VV = [f for f in listdir(sar_inputs_VV) if isfile(os.path.join(sar_inputs_VV, f))]
            # assert len(images_VH) == 2183 and len(images_VH) == len(images_VV)
            # input_names_VH += [os.path.join(sar_inputs_VH, i) for i in images_VH]
            # input_names_VV += [os.path.join(sar_inputs_VV, i) for i in images_VV]
            # gt_names += [os.path.join(os.path.join(dir, 'S2'), i.replace('S1', 'S2')).replace('_VH','').replace('HEB','HEB_1') for i in images_VH]
            #
            # x_VH = list(enumerate(input_names_VH))
            # random.shuffle(x_VH)
            # indices, input_names_VH = zip(*x_VH)
            # input_names_VV = [input_names_VV[idx] for idx in indices]
            # gt_names = [gt_names[idx] for idx in indices]
            self.dir = None
        elif filelist == "syn_val.txt":
            self.dir = None
            input_names, gt_names = [], []
            # SAR val Complex SAR
            # sar_inputs = os.path.join(dir, 'S1')
            # C2Matrix
            sar_inputs = os.path.join(dir, 'C2Matrix')
            # normalized
            # sar_inputs = os.path.join(dir, 'S1_normalize')
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            assert len(images) == 50
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(os.path.join(dir, 'S2'), i.replace('S1', 'S2')) for i in images]
            #
            x = list(enumerate(input_names))
            random.shuffle(x)
            indices, input_names = zip(*x)
            gt_names = [gt_names[idx] for idx in indices]
            # GRD SAR
            # sar_inputs_VH = os.path.join(dir, 'GRD', 'VH')
            # sar_inputs_VV = os.path.join(dir, 'GRD', 'VV')
            # images_VH = [f for f in listdir(sar_inputs_VH) if isfile(os.path.join(sar_inputs_VH, f))]
            # images_VV = [f for f in listdir(sar_inputs_VV) if isfile(os.path.join(sar_inputs_VV, f))]
            # assert len(images_VH) == 50
            # input_names_VH += [os.path.join(sar_inputs_VH, i) for i in images_VH]
            # input_names_VV += [os.path.join(sar_inputs_VV, i) for i in images_VV]
            # gt_names += [os.path.join(os.path.join(dir, 'S2'), i.replace('S1', 'S2')).replace('_VH','').replace('HEB','HEB_1') for i in images_VH]
            #
            # x_VH = list(enumerate(input_names_VH))
            # random.shuffle(x_VH)
            # indices, input_names_VH = zip(*x_VH)
            # input_names_VV = [input_names_VV[idx] for idx in indices]
            # gt_names = [gt_names[idx] for idx in indices]
        elif filelist =="test.txt":
            # 测模拟SAR
            self.dir = None
            # sar_inputs = "/data1/jbwei/Pke/ComplexSLC2S2/test/S1"
            # Complex SAR
            # sar_inputs = "/data1/jbwei/Pke/ComplexSLC2S2/test_patch/S1"
            # C2矩阵
            sar_inputs = "/data1/jbwei/Pke/ComplexSLC2S2/test_patch/C2Matrix"
            # sar_inputs = "/home/jbwei/190_data1/Pke/ComplexSLC2S2/test_patch/C2Matrix"
            # A100
            # sar_inputs = "/home/jbwei/190_data1/Pke/ComplexSLC2S2/test_patch/S1_normalize"
            images = [f for f in listdir(sar_inputs) if isfile(os.path.join(sar_inputs, f))]
            assert len(images) == 7
            input_names += [os.path.join(sar_inputs, i) for i in images]
            gt_names += [os.path.join(sar_inputs.replace('S1', 'S2'), i.replace('S1', 'S2')) for i in images]
            # A100 normalize
            # gt_names += [os.path.join(sar_inputs.replace('S1_normalize', 'S2'), i.replace('S1', 'S2')) for i in images]
            # GRD SAR # v28
            # sar_inputs_VH = "/data1/jbwei/Pke/ComplexSLC2S2/test_patch/GRD/VH"
            # sar_inputs_VV = "/data1/jbwei/Pke/ComplexSLC2S2/test_patch/GRD/VV"
            # gt_inputs = '/data1/jbwei/Pke/ComplexSLC2S2/test_patch/S2'
            # sar_inputs_VH = "/home/jbwei/190_data1/Pke/ComplexSLC2S2/test_patch/GRD/VH"
            # sar_inputs_VV = "/home/jbwei/190_data1/Pke/ComplexSLC2S2/test_patch/GRD/VV"
            # gt_inputs = '/home/jbwei/190_data1/Pke/ComplexSLC2S2/test_patch/S2'
            # images_VH = [f for f in listdir(sar_inputs_VH) if isfile(os.path.join(sar_inputs_VH, f))]
            # images_VV = [f for f in listdir(sar_inputs_VV) if isfile(os.path.join(sar_inputs_VV, f))]
            # assert len(images_VH) == 7 and len(images_VH)==len(images_VV)
            # input_names_VH += [os.path.join(sar_inputs_VH, i) for i in images_VH]
            # input_names_VV += [os.path.join(sar_inputs_VV, i) for i in images_VV]
            # gt_names += [os.path.join(gt_inputs, i.replace('S1', 'S2')).replace('_VH','') for i in images_VH]
        else:
            print("None file")

        self.input_names = input_names
        self.gt_names = gt_names
        self.input_names_VH = input_names_VH
        self.input_names_VV = input_names_VV
        self.patch_size = patch_size # 64
        self.transforms = transforms # ToTensor
        self.n = n # 4
        self.parse_patches = parse_patches # True

    @staticmethod
    def get_params(img, output_size):
        w, h, _ = img.shape  # 256x256 将img.size->img.shape，之前是PIL格式
        x, y = output_size  # 64x64
        return x, y, h, w

    @staticmethod
    def n_random_crops(img, x, y, h, w):
        crops = []
        for i in range(int(h / x)):  # 4
            for j in range(int(w / y)):  # 4
                new_crop = img[i * x: (i + 1) * x, j * y: (j + 1) * y, :]
                crops.append(new_crop)
        return tuple(crops)

    @staticmethod
    def S2Normalize(data):
        mean = [0.5, 0.5, 0.5, 0.5]  # [0.0899, 0.1168, 0.1379, 0.2780]  # 每个通道的均值
        std = [0.5, 0.5, 0.5, 0.5]  # [0.0535, 0.0537, 0.0628, 0.0975]  # 每个通道的标准差
        for i in range(len(mean)):
            data[i] = (data[i] - mean[i]) / std[i]
        return data

        #############随机取块
    # v5
    # @staticmethod
    # def get_params(img, output_size, n):
    #     w, h, _ = img.shape  # 将img.size->img.shape，之前是PIL格式1964,1264
    #     th, tw = output_size
    #     if w == tw and h == th:
    #         return 0, 0, h, w
    #
    #     i_list = [random.randint(0, h - th) for _ in range(n)]  # 1200-x
    #     j_list = [random.randint(0, w - tw) for _ in range(n)]  # 1900-y
    #     return i_list, j_list, th, tw
    #
    # @staticmethod
    # def n_random_crops(img, x, y, h, w):
    #     crops = []
    #     if len(img.shape)==2:
    #         for i in range(len(x)):
    #             new_crop = img[y[i]:y[i] + w,
    #                         x[i]:x[i] + h]  # img.crop((y[i], x[i], y[i] + w, x[i] + h))#PIL库的crop()函数用于裁剪图片
    #             crops.append(new_crop)
    #     else:
    #         for i in range(len(x)):
    #             new_crop = img[y[i]:y[i] + w,
    #                         x[i]:x[i] + h,:]  # img.crop((y[i], x[i], y[i] + w, x[i] + h))#PIL库的crop()函数用于裁剪图片
    #             crops.append(new_crop)
    #     return tuple(crops)

    def get_images(self, index):
        # Complex SAR
        input_name = self.input_names[index]
        gt_name = self.gt_names[index]
        # print(re.split('/', input_name)[-1][:-4])
        # print(re.split('/', gt_name)[-1][:-4])
        assert re.split('/', input_name)[-1][:-4] == re.split('/', gt_name)[-1][:-4].replace('S2', 'S1')
        img_id = re.split('/', input_name)[-1][:-4]
        # 将输入读取改为sar图像tif格式，使用imageio
        input_img = imageio.v3.imread(os.path.join(self.dir, input_name)) if self.dir else imageio.v3.imread(input_name)
        gt_img = imageio.v3.imread(os.path.join(self.dir, gt_name)) if self.dir else imageio.v3.imread(gt_name)
        # GRD SAR
        # input_VH_name = self.input_names_VH[index]
        # input_VV_name = self.input_names_VV[index]
        # gt_name = self.gt_names[index]
        # print(input_VH_name, gt_name)
        # GRD
        # assert re.split('/', input_VH_name)[-1][:-4].replace('_VH', '') == re.split('/', gt_name)[-1][:-4].replace('S2', 'S1').replace('HEB_1','HEB')
        # SLC
        assert re.split('/', input_name)[-1][:-4].replace('_VH', '') == re.split('/', gt_name)[-1][:-4].replace('S2', 'S1')
        # img_id = re.split('/', input_VH_name)[-1][:-4].replace('_VH', '')
        # input_img_VH = imageio.v3.imread(os.path.join(self.dir, input_VH_name)) if self.dir else imageio.v3.imread(input_VH_name)
        # input_img_VV = imageio.v3.imread(os.path.join(self.dir, input_VV_name)) if self.dir else imageio.v3.imread(input_VV_name)
        # input_img = np.stack((input_img_VH, input_img_VV), axis=2)
        # gt_img = imageio.v3.imread(os.path.join(self.dir, gt_name)) if self.dir else imageio.v3.imread(gt_name)

        # print(gt_img.dtype,gt_img.shape)
        # noise, lambda_= utils.gamma_noise_v2(input_img, Look=1)
        # noise = noise[0]
        # lambda_ = noise[1]
        # gt_img = utils.x0_process_v2(gt_img, lambda_)
        # print(gamma_noise.dtype,gamma_noise.shape)
        # gt_img_VV = imageio.v3.imread(os.path.join(self.dir, gt_name_VV)) if self.dir else imageio.v3.imread(gt_name_VV)
        # gt_img = Image.open(os.path.join(self.dir, gt_name)) if self.dir else Image.open(gt_name)
        # gt_img = gt_img.convert('L')
        # 作为float32输入
        # gt_img = np.array(gt_img)
        # C2Matrix
        input_img = input_img.astype(np.float32)
        # # print(np.max(input_img), np.min(input_img))
        # if input_img.shape[0] == self.input_channel:
        #     input_img = np.transpose(input_img, (1,2,0))
        gt_img = (gt_img / 10000.0).astype(np.float32) # (256, 256, 4)
        # # print(np.max(gt_img), np.min(gt_img))
        # GRD
        # if gt_img.shape[0] == self.input_channel * 2:
        # Complex SAR
        if gt_img.shape[0] == self.input_channel:
            gt_img = np.transpose(gt_img, (1,2,0))
        # print(np.min(input_img),np.max(input_img))
        # print(np.min(gt_img),np.max(gt_img))
        # v2 v3 v4 v5 v6
        # gt_img = self.S2Normalize(gt_img)
        # print('gt_img', gt_img.shape)
        # gt_img_VV = gt_img_VV.astype(np.float32)

        if self.parse_patches:
            # 随机裁剪
            # x, y, h, w = self.get_params(input_img, (self.patch_size, self.patch_size), 4)
            # 固定裁剪
            x, y, h, w = self.get_params(input_img, (self.patch_size, self.patch_size))
            input_img = self.n_random_crops(input_img, x, y, h, w) # 36 64x64
            # input_img_VV = self.n_random_crops(input_img_VV, x, y, h, w) # 36 64x64
            gt_img = self.n_random_crops(gt_img, x, y, h, w) # 36 64x64
            # gamma_noise = self.n_random_crops(noise, x, y, h, w) # 36 64x64 *
            # print(gamma_noise.shape)
            # gt_img_VV = self.n_random_crops(gt_img_VV, x, y, h, w) # 36 64x64
            outputs = [torch.cat(
                    [self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
                        for i in range(self.n)]
            # outputs = [torch.cat(
            #         [self.transforms(input_img[i]), self.transforms(gt_img[i])], dim=0)
            #             for i in range(self.n)]
            return torch.stack(outputs, dim=0), img_id
        else:
            # Resizing images to multiples of 16 for whole-image restoration
            # wd_new, ht_new = input_img.shape
            # if ht_new > wd_new and ht_new > 1024:
            #     wd_new = int(np.ceil(wd_new * 1024 / ht_new))
            #     ht_new = 1024
            # elif ht_new <= wd_new and wd_new > 1024:
            #     ht_new = int(np.ceil(ht_new * 1024 / wd_new))
            #     wd_new = 1024
            # wd_new = int(16 * np.ceil(wd_new / 16.0))
            # ht_new = int(16 * np.ceil(ht_new / 16.0))
            # input_img.resize((wd_new, ht_new))
            # gt_img.resize((wd_new, ht_new))
            # gamma_noise = utils.gamma_noise_v2(input_img)
            # gamma_noise = gamma_noise[0]
            # lambda_ = gamma_noise[1]
            # print(gamma_noise.shape)
            # print(input_img.shape, gt_img.shape)

            return torch.cat([self.transforms(input_img), self.transforms(gt_img)], dim=0), img_id

    def __getitem__(self, index):
        res = self.get_images(index)
        return res

    def __len__(self):
        # Complex SAR
        return len(self.input_names)
        # GRD SAR
        # return len(self.input_names_VH)
