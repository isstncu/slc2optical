import datetime
from math import log as ln
import os
import time
from inspect import isfunction

import torchvision.transforms
from torchvision.transforms.functional import crop

import utils
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import utils.metrics
import utils.gamma_noise
from models.unet import DiffusionUNet
from models.unet_CDS import DiffusionComplexUNet_CDS
from models.unet_complex import DiffusionComplexUNet
from models.unet_complex_end_to_end import DiffusionComplexUNet_End_to_End
from utils.loss import discretized_gaussian_log_likelihood, Loss

import warnings
warnings.filterwarnings('ignore')

# This script is adapted from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

# def data_transform(X):
#     return 2 * X - 1.0

# def inverse_data_transform(X):
#     return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

class EMAHelper(object):
    def __init__(self, mu=0.9999):
        self.mu = mu
        self.shadow = {}

    def register(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                self.shadow[name].data = (1. - self.mu) * param.data + self.mu * self.shadow[name].data

    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        for name, param in module.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        self.ema(module_copy)
        return module_copy

    def state_dict(self):
        return self.shadow

    def load_state_dict(self, state_dict):
        self.shadow = state_dict

def get_beta_schedule(beta_schedule, *, beta_start, beta_end, num_diffusion_timesteps):
    def sigmoid(x):
        return 1 / (np.exp(-x) + 1)

    if beta_schedule == "quad":
        betas = (np.linspace(beta_start ** 0.5, beta_end ** 0.5, num_diffusion_timesteps, dtype=np.float64) ** 2)
    elif beta_schedule == "linear":
        betas = np.linspace(beta_start, beta_end, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "const":
        betas = beta_end * np.ones(num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "jsd":  # 1/T, 1/(T-1), 1/(T-2), ..., 1
        betas = 1.0 / np.linspace(num_diffusion_timesteps, 1, num_diffusion_timesteps, dtype=np.float64)
    elif beta_schedule == "sigmoid":
        betas = np.linspace(-6, 6, num_diffusion_timesteps)
        betas = sigmoid(betas) * (beta_end - beta_start) + beta_start
    else:
        raise NotImplementedError(beta_schedule)
    assert betas.shape == (num_diffusion_timesteps,)
    return betas


def noise_estimation_loss(model, x0, t, e, b):
    a = (1-b).cumprod(dim=0).index_select(0, t).view(-1, 1, 1, 1)
    # Complex SAR
    x = x0[:, 4:, :, :] * a.sqrt() + e * (1.0 - a).sqrt() # 加高斯噪声
    # Complex SAR
    output = model(torch.cat([x0[:, :4, :, :], x], dim=1), t.float()) # cat input 与 x_noisy
    # GRD SAR
    # x = x0[:, 2:, :, :] * a.sqrt() + e * (1.0 - a).sqrt()  # 加高斯噪声
    # GRD SAR
    # output = model(torch.cat([x0[:, :2, :, :], x], dim=1), t.float()) # cat input 与 x_noisy
    # FFT
    output_fft = torch.fft.fftn(output, dim=(-2, -1))
    e_fft = torch.fft.fftn(e, dim=(-2, -1))
    mse_fft = (e_fft - output_fft).abs().sum(dim=(1, 2, 3)).mean(dim=0)

    # train loss: mse + tv_loss + fft_mse
    # tv_loss = utils.loss.TVLoss()
    # lambda_tv = 10
    # MSE
    # print('MSE:{0:.2f}'.format((e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)))
    # return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)
    # MSE + FFT_MSE
    print('MSE:{0:.2f}'.format((e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)), 'FFT_MSE:{0:.2f}'.format(mse_fft))
    fft_weight = 0.001
    return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0) + fft_weight * mse_fft


class DenoisingDiffusion(object):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.config = config
        self.device = config.device


        # v1 v2 v4 v5
        # self.model = DiffusionUNet(config)
        # v3 v14
        # self.model = DiffusionComplexUNet(self.config.model.in_channels, self.config.model.out_ch)
        # End to end
        self.model = DiffusionComplexUNet_End_to_End(self.config.model.in_channels, self.config.model.out_ch)
        # CDS
        # self.model = DiffusionComplexUNet_CDS(self.config.model.in_channels, self.config.model.out_ch)
        self.model.to(self.device)
        self.model = torch.nn.DataParallel(self.model)

        self.ema_helper = EMAHelper()
        self.ema_helper.register(self.model)

        self.optimizer = utils.optimize.get_optimizer(self.config, self.model.parameters())
        self.start_epoch, self.step = 0, 0
        self.NAT = 1. / ln(2.0)
        self.mse = nn.MSELoss()

        betas = get_beta_schedule(
            beta_schedule=config.diffusion.beta_schedule,
            beta_start=config.diffusion.beta_start,
            beta_end=config.diffusion.beta_end,
            num_diffusion_timesteps=config.diffusion.num_diffusion_timesteps,
        )

        betas = self.betas = torch.from_numpy(betas).float().to(self.device)
        self.num_timesteps = betas.shape[0]


    #IDDPM-functions-losses
    def extract(self, arr, timesteps, broadcast_shape):
        """
        Extract values from a 1-D numpy array for a batch of indices.

        :param arr: the 1-D tensor.
        :param timesteps: a tensor of indices into the array to extract.
        :param broadcast_shape: a larger shape of K dimensions with the batch
                                dimension equal to the length of timesteps.
        :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
        """
        res = arr[timesteps].float()
        while len(res.shape) < len(broadcast_shape):
            res = res[..., None]
        res = res.expand(broadcast_shape)
        return res

    def unnormalize_to_zero_to_one(self, t):
        return (t + 1) * 0.5

    def normal_kl(self, mean1, logvar1, mean2, logvar2):
        """
        KL divergence between normal distributions parameterized by mean and log-variance.
        由均值和对数方差参数化的正态分布之间的KL散度。
        """
        return 0.5 * (-1.0 + logvar2 - logvar1 + torch.exp(logvar1 - logvar2) + ((mean1 - mean2) ** 2) * torch.exp(
            -logvar2))

    def meanflat(self, x):
        return x.mean(dim=list(range(1, len(x.shape))))

    # def gamma_estimation_loss(self, model,
    #                           x: torch.Tensor,
    #                           t: torch.LongTensor,
    #                           b: torch.Tensor,
    #                           theta_0=0.001,
    #                           keepdim=False):
    #     x0 = x[:, 1:, :, :]
    #     a = (1 - b).cumprod(dim=0) # a: torch.Size([1000])
    #     k = (b / a) / theta_0 ** 2 # k: torch.Size([1000])
    #     theta = (a.sqrt() * theta_0).index_select(0, t).view(-1, 1, 1, 1) # theta: torch.Size([20, 1, 1, 1])
    #     k_bar = k.cumsum(dim=0).index_select(0, t).view(-1, 1, 1, 1) # k_bar: torch.Size([20, 1, 1, 1])
    #     a = a.index_select(0, t).view(-1, 1, 1, 1) # a: torch.Size([20, 1, 1, 1])
    #     concentration = torch.ones(x0.size()).to(x0.device) * k_bar # concentration: torch.Size([20, 1, 128, 128])
    #     rates = torch.ones(x0.size()).to(x0.device) * theta # rates: torch.Size([20, 1, 128, 128])
    #     m = torch.distributions.Gamma(concentration, 1 / rates) # Gamma object
    #     e = m.sample() # torch.Size([20, 1, 128, 128])
    #     e = e - concentration * rates
    #     e = e / (1.0 - a).sqrt()
    #     x_t = a.sqrt() * x0 + e * (1.0 - a).sqrt()
    #     output = model(torch.cat([x[:, :1, :, :], x_t], dim=1), t.float())
    #     # print('predict:', torch.max(output), torch.min(output))
    #     # print('e:', torch.max(e), torch.min(e))
    #     if keepdim:
    #         return (e - output).square().sum(dim=(1, 2, 3))
    #     else:
    #         return (e - output).square().sum(dim=(1, 2, 3)).mean(dim=0)

    def p_losses(self, model, x, t, noise, betas, clip_denoised=True):

        # noise = default(noise, lambda: torch.randn_like(x0))  #noise=None
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        assert alphas_cumprod_prev.shape == (self.num_timesteps,)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

        # q_sample
        x0 = x[:, 4:, :, :]
        # x0 = utils.process(x0).to(self.device)
        x_t = (self.extract(sqrt_alphas_cumprod, t, x0.shape) * x0
               + self.extract(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)
        # print(x_t.shape)
        #utils.logging.save_image(x_t, os.path.join('results/images/SAR/Synthesis_no_log_SAR/v29/x_t_output.tif'))

        # model output
        # temp = utils.process(x[:, :1, :, :])
        model_output = model(torch.cat([x[:, :4, :, :], x_t], dim=1),
                             t.float())

        # calculating kl loss for learned variance (interpolation)
        # q_posterior: 计算扩散后验的均值和方差 q(x_{t-1} | x_t, x_0)
        posterior_mean_coef1 = betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_mean_coef2 = (1. - alphas_cumprod_prev) * torch.sqrt(alphas) / (1. - alphas_cumprod)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        posterior_log_variance_clipped = torch.log(posterior_variance.clamp(min=1e-20))
        # 后验均值和方差
        true_mean = (self.extract(posterior_mean_coef1, t, x_t.shape) * x0
                     + self.extract(posterior_mean_coef2, t, x_t.shape) * x_t)
        true_log_variance_clipped = self.extract(posterior_log_variance_clipped, t, x_t.shape)

        # p_mean_variance: 传入的是x_t,预测t−1时刻的 均值mean和方差variance
        sqrt_recip_alphas_cumprod = torch.sqrt(1. / alphas_cumprod)
        sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / alphas_cumprod - 1)
        pred_noise, var_interp_frac_unnormalized = model_output.chunk(2, dim=1)
        # print(pred_noise.shape)

        # predict_start_from_noise
        pred_x_start = (self.extract(sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                   - self.extract(sqrt_recipm1_alphas_cumprod, t, x_t.shape) * pred_noise)
        if clip_denoised:
            pred_x_start.clamp_(-1., 1.)
        var_interp_frac = self.unnormalize_to_zero_to_one(var_interp_frac_unnormalized)
        max_log = self.extract(torch.log(betas), t, x_t.shape)
        min_log = self.extract(posterior_log_variance_clipped, t, x_t.shape)
        model_mean = (self.extract(posterior_mean_coef1, t, x_t.shape) * pred_x_start
                      + self.extract(posterior_mean_coef2, t, x_t.shape) * x_t)
        model_log_variance = var_interp_frac * max_log + (1 - var_interp_frac) * min_log

        # kl loss with detached model predicted mean, for stability reasons as in paper
        detached_model_mean = model_mean.detach()
        kl = self.normal_kl(true_mean, true_log_variance_clipped, detached_model_mean, model_log_variance)
        kl = self.meanflat(kl) * self.NAT
        decoder_nll = -discretized_gaussian_log_likelihood(
            x0, means=detached_model_mean, log_scales=0.5 * model_log_variance
        )
        assert decoder_nll.shape == x0.shape
        decoder_nll = self.meanflat(decoder_nll) * self.NAT
        # at the first timestep return the decoder NLL, otherwise return KL(q(x_{t-1}|x_t,x_0) || p(x_{t-1}|x_t))
        kl_losses = torch.where(t == 0, decoder_nll, kl).mean(dim=0)
        # simple loss - predicting noise, x0, or x_prev
        kl_loss_weight = 64*64
        #mse_losses = self.mse(noise, pred_noise)
        #print(pred_noise.shape)
        # length = pred_noise.shape[0]
        # mse = torch.nn.MSELoss()
        # print('predict:', torch.max(pred_noise).item(), torch.min(pred_noise).item())
        # print('e:', torch.max(noise).item(), torch.min(noise).item())
        mse_losses = (noise - pred_noise).square().sum(dim=(1, 2, 3)).mean(dim=0)# 16x1x64x64的16张图片mse的平均值，不是每个像素mse的平均值
        # mse_losses = self.mse(noise, pred_noise)
        print('MSE:{0:.6f}, KL:{1:.6f}'.format(mse_losses, kl_losses * kl_loss_weight))
        return mse_losses + kl_loss_weight * kl_losses

    def load_ddm_ckpt(self, load_path, ema=False):
        checkpoint = utils.logging.load_checkpoint(load_path, None)
        self.start_epoch = checkpoint['epoch']
        self.step = checkpoint['step']
        self.model.load_state_dict(checkpoint['state_dict'], strict=True)
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.ema_helper.load_state_dict(checkpoint['ema_helper'])
        if ema:
            self.ema_helper.ema(self.model)
        print("=> loaded checkpoint '{}' (epoch {}, step {})".format(load_path, checkpoint['epoch'], self.step))

    def train(self, DATASET):
        cudnn.benchmark = True
        train_loader, val_loader = DATASET.get_loaders()

        if os.path.isfile(self.args.resume):
            self.load_ddm_ckpt(self.args.resume)

        least_loss = float('inf')
        for epoch in range(self.start_epoch, self.config.training.n_epochs):
            print('epoch: ', epoch)
            data_start = time.time()
            data_time = 0
            for i, (x, y) in enumerate(train_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                n = x.size(0)
                # print(x.shape) # (16,6,128,128) 或者(64,8,64,64)
                data_time += time.time() - data_start
                self.model.train()
                self.step += 1

                x = x.to(self.device)
                # x = data_transform(x)
                # Complex SAR
                noise = torch.randn_like(x[:, 4:, :, :])
                # GRD SAR
                # noise = torch.randn_like(x[:, 2:, :, :])
                # noise = x[:, 2:, :, :]
                # noise = utils.gamma_noise(x[:, 1:, :, :]).to(self.device)
                b = self.betas


                # antithetic sampling
                t = torch.randint(low=0, high=self.num_timesteps, size=(n // 2 + 1,)).to(self.device)
                t = torch.cat([t, self.num_timesteps - t - 1], dim=0)[:n]
                loss = noise_estimation_loss(self.model, x, t, noise, b) # 1.损失函数可以换成DG(despeckling gain),优于MSE和MAE
                # loss = self.p_losses(self.model, x, t, noise, b)
                # print('MSE:{0:.6f}'.format(loss))

                if self.step % 10 == 0:
                    print(f"{self.config.training.version}, epoch: {epoch} step: {self.step}, loss: {loss.item():.4f}, data time: {data_time / (i+1)}")

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema_helper.update(self.model)
                data_start = time.time()

                # 验证集与验证损失
                if self.step % self.config.training.validation_freq == 0:
                    # v15
                    loss_function = Loss(self.device, 2e-3) # DG + TV loss ID-CNN中λ=2e-3
                    # v1-v14
                    # loss_function = nn.MSELoss()
                    self.model.eval()
                    val_loss, ssim, psnr = self.sample_validation_patches(val_loader, self.step, loss_function)
                    print("validating->ssim:{:.4f}, psnr:{:.4f}, loss:{:.4f}".format(ssim, psnr, val_loss))
                    if not os.path.exists(os.path.join('./ckpts', self.args.model_name, self.config.training.version)):
                        os.makedirs(os.path.join('./ckpts', self.args.model_name, self.config.training.version))
                    with open(os.path.join('./ckpts', self.args.model_name, self.config.training.version, '{}_{}.txt'.format(self.config.training.name, self.config.training.version)), mode="a", encoding="utf-8") as f:
                        f.write(f"step: {self.step}, ssim: {ssim:.4f}, psnr: {psnr:.4f}, val_loss: {val_loss}, least_error: {least_loss}, time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                        f.write("\n")
                    f.close()

                # if self.step % self.config.training.snapshot_freq == 0 or self.step == 1:
                    if val_loss < least_loss:
                        least_loss = val_loss
                        with open(os.path.join('./ckpts', self.args.model_name, self.config.training.version, '{}_{}.txt'.format(self.config.training.name, self.config.training.version)), mode="a", encoding="utf-8") as f:
                            f.write(
                                f"step: {self.step}, least_error: {least_loss}, saving checkpoint!")
                            f.write("\n")
                        f.close()
                        utils.logging.save_checkpoint({
                            'epoch': epoch + 1,
                            'step': self.step,
                            'state_dict': self.model.state_dict(),
                            'optimizer': self.optimizer.state_dict(),
                            'ema_helper': self.ema_helper.state_dict(),
                            'params': self.args,
                            'config': self.config
                        }, filename=os.path.join(
                            './ckpts', self.args.model_name, self.config.training.version, '{}_ddpm_best_epoch_{}'.format(self.config.training.name, epoch+1)
                        ))
            if (epoch+1-200) % 100 == 0:
                utils.logging.save_checkpoint({
                    'epoch': epoch + 1,
                    'step': self.step,
                    'state_dict': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict(),
                    'ema_helper': self.ema_helper.state_dict(),
                    'params': self.args,
                    'config': self.config
                }, filename=os.path.join(
                    './ckpts', self.args.model_name, self.config.training.version,
                    '{}_ddpm_epoch_{}'.format(self.config.training.name, epoch+1)
                ))
            # 2 & 2_val:norm(log(x0))+norm(Y(log(gamma)))
            # 4:norm(log(x0)) + gaussian
            # if epoch+1 >= 300 and epoch+1 % 25 ==0:
            #     utils.logging.save_checkpoint({
            #         'epoch': epoch + 1,
            #         'step': self.step,
            #         'state_dict': self.model.state_dict(),
            #         'optimizer': self.optimizer.state_dict(),
            #         'ema_helper': self.ema_helper.state_dict(),
            #         'params': self.args,
            #         'config': self.config
            #     }, filename=os.path.join(
            #         self.config.data.data_dir, 'ckpts',
            #         '{}_epoch{}_ddpm'.format(self.config.training.name, epoch+1)
            #     ))

    # ddim
    def sample_image(self, x_cond, x, last=True, patch_locs=None, patch_size=None):
        if self.config.sampling.sample_type == "ddim":
            skip = self.config.diffusion.num_diffusion_timesteps // self.args.sampling_timesteps # 1000 // 25 = 40
            seq = range(0, self.config.diffusion.num_diffusion_timesteps, skip) # range(0,1000,40)
            if patch_locs is not None:
                xs = utils.sampling.generalized_steps_overlapping(x, x_cond, seq, self.model, self.betas, eta=0.,
                                                                  corners=patch_locs, p_size=patch_size)
            else:
                xs = utils.sampling.generalized_steps(x, x_cond, seq, self.model, self.betas, eta=0.)
            if last:
                return xs[0][-1]
                # h_list, w_list = self.overlapping_grid_indices(x_cond, output_size=32, r=2)
                # patch_locs = [(i, j) for i in h_list for j in w_list]
                # model_file = '/home/jbwei/Pke/ResidualAttentionNetwork-pytorch/Residual-Attention-Network/model_92_sgd_best_SLC_mean_92.pkl'
                # pred_model.load_state_dict(torch.load(model_file))
                # xs = xs[0][-1]
                # # 均值修正结果
                # x_grid_mask = torch.zeros_like(x, device=x.device)
                # # print(corners)
                # for (hi, wi) in patch_locs:
                #     x_grid_mask[:, :, hi:hi + 32, wi:wi + 32] += 1
                #
                # manual_batching_size = 36 # 32 # 64
                # xs_patch = torch.cat([crop(xs, hi, wi, 32, 32) for (hi, wi) in patch_locs], dim=0)
                # x_cond_patch = torch.cat([crop(x_cond, hi, wi, 32, 32) for (hi, wi) in patch_locs], dim=0)
                # xt_next_mask = torch.zeros_like(xs)
                # # x_cond_repeat = x_cond.repeat(2, 1, 1, 1)  # torch.Size([2, 4, 1024, 1024])
                # # print(x_cond.shape)
                # # print(len(patch_locs))
                # for i in range(0, len(patch_locs), manual_batching_size):  # len(corners) = 1369
                #     xs_patch_mean = torch.mean(xs_patch[i:i+manual_batching_size], dim=(2, 3), keepdim=True)
                #     print(x_cond_patch[i:i+manual_batching_size].shape)
                #     pred_patch_mean = pred_model(x_cond_patch[i:i+manual_batching_size])
                #     xs_patch_temp = (xs_patch[i:i+manual_batching_size] / xs_patch_mean) * pred_patch_mean
                #     # print(xs_patch_temp.shape)
                #     for idx, (hi, wi) in enumerate(patch_locs[i:i + manual_batching_size]):
                #         xt_next_mask[0, :, hi:hi + 32, wi:wi + 32] += xs_patch_temp[idx]
                # xt_next_mask = torch.div(xt_next_mask, x_grid_mask)
                # for i in range(32):
                #     for j in range(32):
                #         xt_next_patch = xs[:, :, i * 32: (i + 1) * 32, j * 32: (j + 1) * 32]
                #         xt_patch_mean = torch.mean(xt_next_patch, dim=(2, 3), keepdim=True)
                #         # print(x_cond[:,:,i * 32: (i + 1) * 32, j * 32: (j + 1) * 32].shape) # torch.Size([1, 4, 32, 32])
                #         pred_mean = pred_model(x_cond_repeat[:, :, i * 32: (i + 1) * 32, j * 32: (j + 1) * 32])
                #         # print(pred_mean.shape)
                #         xt_next_patch = (xt_next_patch / xt_patch_mean) * pred_mean[0]
                #         xt_next_mask[:, :, i * 32: (i + 1) * 32, j * 32: (j + 1) * 32] += xt_next_patch
            # return xt_next_mask.to('cpu')
            return xs
        else:
            raise NotImplementedError

    def sample_validation_patches(self, val_loader, step, loss_function):
        # image_folder = os.path.join(self.args.image_folder, self.config.training.name + self.config.training.version)#results/images/SAR64
        image_folder = os.path.join('./validation', self.config.training.version)
        if not os.path.exists(image_folder):
            os.makedirs(image_folder)
        with torch.no_grad():
            print(f"Processing a single batch of validation images at step: {step}")
            for i, (x, y) in enumerate(val_loader):
                x = x.flatten(start_dim=0, end_dim=1) if x.ndim == 5 else x
                break
            n = x.size(0)
            x_cond = x[:, :self.config.data.channels, :, :].to(self.device)
            target = x[:, self.config.data.channels:, :, :].to(self.device)
            # x_cond = data_transform(x_cond)
            x = torch.randn(n, target.shape[1], self.config.data.image_size, self.config.data.image_size, device=self.device)
            # x = x[:, 2:, :, :].to(self.device)
            # x = x[0].to(self.device)
            # lambda_ = x[1]

            x = self.sample_image(x_cond, x).to(self.device)
            val_loss = loss_function(x, target)

            ssim, psnr = 0, 0
            # ssim = utils.metrics.calculate_ssim(x.unsqueeze(0), target.unsqueeze(0))
            # psnr = utils.metrics.calculate_psnr(x.unsqueeze(0), target.unsqueeze(0))
            for i in range(n):
                cal_ssim = utils.metrics.calculate_ssim(x[i].unsqueeze(0), target[i].unsqueeze(0))
                cal_psnr = utils.metrics.calculate_psnr(x[i].unsqueeze(0), target[i].unsqueeze(0))
                if cal_ssim > ssim:
                    ssim = cal_ssim
                if cal_psnr > psnr:
                    psnr = cal_psnr
                # utils.logging.save_image(target[i], os.path.join(image_folder, str(step), f"{y[int(i/16)]}_{int(i/16)*16+i%16}_target.tif"))
                # utils.logging.save_image(x[i], os.path.join(image_folder, str(step), f"{y[int(i/16)]}_{int(i/16)*16+i%16}.tif"))

        return val_loss, ssim, psnr

    def overlapping_grid_indices(self, x_cond, output_size, r=None):
        _, c, h, w = x_cond.shape
        r = 16 if r is None else r
        h_list = [i for i in range(0, h - output_size + 1, r)]  # [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576]
        w_list = [i for i in range(0, w - output_size + 1, r)]  # [0, 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352, 368, 384, 400, 416, 432, 448, 464, 480, 496, 512, 528, 544, 560, 576]
        return h_list, w_list
