import math
import complexPyTorch
from complexPyTorch.complexLayers import ComplexBatchNorm2d, ComplexConv2d, ComplexLinear, ComplexDropout2d
from complexPyTorch.complexFunctions import complex_relu, complex_max_pool2d, complex_dropout, complex_avg_pool2d
import torch
import torch.nn as nn
import models.layers_CDS as layers
import utils


def get_timestep_embedding(timesteps, embedding_dim):
    """
    This matches the implementation in Denoising Diffusion Probabilistic Models:
    From Fairseq.
    Build sinusoidal embeddings.
    This matches the implementation in tensor2tensor, but differs slightly
    from the description in Section 3.5 of "Attention Is All You Need".
    """
    assert len(timesteps.shape) == 1

    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
    emb = emb.to(device=timesteps.device)
    emb = timesteps.float()[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
    return emb


def nonlinearity(x):
    # swish SiLU()
    return x * torch.sigmoid(x)
    #relu = nn.ReLU(inplace=True)
    #return relu(x)


def nonlinearity_complex(x):
    # swish SiLU()
    return complex_relu(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)
    #return nn.BatchNorm2d(in_channels)


def Normalize_complex(in_channels):
    return ComplexBatchNorm2d(in_channels)  # , track_running_stats = False)#track_running_stats默认 True


def zero_module(module):
    """
    Zero out the parameters of a module and return it.将模块的参数归零并返回
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        # self.UP = nn.Sequential(
        #     nn.Conv2d(in_channels, 4 * in_channels, 3, stride=1, padding=1),
        #     nn.PixelShuffle(2)
        # )
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        #x = self.UP(x)
        if self.with_conv:
            x = self.conv(x)
        return x


class Downsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=2,
                                        padding=0)

    def forward(self, x):
        if self.with_conv:  # default true
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x


class Downsample_complex(nn.Module):
    def __init__(self, in_channels, with_conv=False, groups=1, diff=False):
        super().__init__()
        self.with_conv = with_conv
        self.with_diff = diff
        if self.with_conv:
            # no asymmetric padding in torch conv, must do it ourselves
            # self.conv = ComplexConv2d(in_channels,
            #                           in_channels,
            #                           kernel_size=3,
            #                           stride=2,
            #                           padding=1)
            self.conv = layers.ComplexConv(in_channels, in_channels, kern_size=3, stride=(
            2, 2), reflect=1, groups=groups, new_init=True, use_groups_init=True)
        if self.with_diff:
            self.diff = layers.DivLayer(in_channels, 3, reflect=1, new_init=True)

    def forward(self, x):
        if self.with_conv:  # default true
            # pad = (0, 1, 0, 1)
            # x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            if self.with_diff:
                x = self.diff(x)
        else:
            x = complex_avg_pool2d(x, kernel_size=2, stride=2)
        return x


class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,
                                     out_channels,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1)
        self.temb_proj = torch.nn.Linear(temb_channels,
                                         out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)
        # self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = zero_module(torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))
        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = torch.nn.Conv2d(in_channels,
                                                     out_channels,
                                                     kernel_size=3,
                                                     stride=1,
                                                     padding=1)
            else:
                self.nin_shortcut = torch.nn.Conv2d(in_channels,
                                                    out_channels,
                                                    kernel_size=1,
                                                    stride=1,
                                                    padding=0)

    def forward(self, x, temb):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)
        #if temb is not None:
        h = h + self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)
        return x + h


class ResnetBlock_complex(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut
        self.gtrelu1 = layers.GTReLU(in_channels, phase_scale=True)
        self.gtrelu2 = layers.GTReLU(out_channels, phase_scale=True)
        self.norm1 = Normalize_complex(in_channels)
        self.conv1 = ComplexConv2d(in_channels,
                                   out_channels,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)

        self.norm2 = Normalize_complex(out_channels)
        self.dropout = ComplexDropout2d(dropout)

        # self.conv2 = ComplexConv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)
        self.conv2 = zero_module(ComplexConv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1))

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                self.conv_shortcut = ComplexConv2d(in_channels,
                                                   out_channels,
                                                   kernel_size=3,
                                                   stride=1,
                                                   padding=1)
            else:
                self.nin_shortcut = ComplexConv2d(in_channels,
                                                  out_channels,
                                                  kernel_size=1,
                                                  stride=1,
                                                  padding=0)

    def forward(self, x):  ##, temb):
        h = x
        h = self.norm1(h)
        # h = nonlinearity_complex(h)
        h = torch.stack([h.real, h.imag], dim=1)
        h = self.gtrelu1(h)
        h = torch.complex(h[:, 0, :, :, :], h[:, 1, :, :, :]).to(torch.complex64)
        h = self.conv1(h)

        h = self.norm2(h)
        h = torch.stack([h.real, h.imag], dim=1)
        # h = nonlinearity_complex(h)
        h = self.gtrelu2(h)
        h = torch.complex(h[:, 0, :, :, :], h[:, 1, :, :, :]).to(torch.complex64)
        # h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x + h


class AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        # self.proj_out = torch.nn.Conv2d(in_channels,in_channels,kernel_size=1,stride=1,padding=0)
        self.proj_out = zero_module(torch.nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, padding=0))

    def forward(self, x):
        # print(x.shape)
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h * w)
        q = q.permute(0, 2, 1)  # b,hw,c
        k = k.reshape(b, c, h * w)  # b,c,hw
        w_ = torch.bmm(q, k)  # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c) ** (-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h * w)
        w_ = w_.permute(0, 2, 1)  # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)
        return x + h_

# class CDS_I(nn.Module):
#     """
#     CDS Model (I-Type) for small CIFAR experiments. Based on CIFARnet
#     """
#     # Our model for the real-valued dataset experiments
#
#     def __init__(self, cifarnet_config='dgtf', dset_type='lab', outsize=10, prototype_size=128, *args, **kwargs):
#         super(CDS_I, self).__init__()
#         self.cifarnet_config = cifarnet_config
#
#         # Building layers....
#         conv = layers.ComplexConv
#         diff = layers.DivLayer
#
#         self.wfm1 = conv(1, 16, kern_size=3, stride=(
#             2, 2), reflect=1, new_init=True, use_groups_init=True, bias=False)
#
#         self.wfm2 = conv(16, 32, kern_size=3,
#                          stride=(2, 2), reflect=1, groups=2, new_init=True, use_groups_init=True)
#         self.wfm3 = conv(32, 64, kern_size=3,
#                          stride=(2, 2), reflect=1, groups=4, new_init=True, use_groups_init=True)
#         self.wfm4 = conv(64, 64, kern_size=4, groups=64,
#                          new_init=True, use_groups_init=True)
#
#         self.diff1 = diff(16, 3, reflect=1, new_init=True)
#
#         self.gtrelu1 = layers.GTReLU(16, phase_scale=True)
#         self.gtrelu2 = layers.GTReLU(32, phase_scale=True)
#         self.gtrelu3 = layers.GTReLU(64, phase_scale=True)
#
#         self.fc1 = conv(64, prototype_size, 1, groups=4, new_init=True)
#
#         self.bn = nn.BatchNorm1d(prototype_size*2)
#
#         dist_feat = layers.DistFeatures
#
#         self.dist_feat = dist_feat(prototype_size, outsize)
#
#     def forward(self, x):
#         # Convert complex input into a real-imaginary input
#         x = torch.stack([x.real, x.imag], dim=1)
#         x = self.wfm1(x)
#         x = self.diff1(x)
#         x = self.gtrelu1(x)
#         x = self.wfm2(x)
#         x = self.gtrelu2(x)
#         x = self.wfm3(x)
#         x = self.gtrelu3(x)
#         x = self.wfm4(x)
#
#         x = self.fc1(x)
#         x_shape = x.shape
#         x = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape)
#         x = self.dist_feat(x[..., 0, 0])
#
#         return x

class DiffusionComplexUNet_CDS(nn.Module):
    def __init__(self, input_nc, output_nc):
        super().__init__()
        ch, out_ch, ch_mult = 128, output_nc, (1,1,2,3,4)# tuple([1,2,3,4])  #[1,1,2,2,4,4]  [1,2,3,4]
        num_res_blocks = 2
        attn_resolutions = (16, 8) # [16,8]
        dropout = 0.0
        in_ch = input_nc
        in_channels = in_ch #+ out_ch  # 2+4=6
        resolution = 64 # 64
        resamp_with_conv = True

        self.ch = ch  # 128
        self.temb_ch = self.ch * 4  # 512
        self.num_resolutions = len(ch_mult)  # 5
        self.num_res_blocks = num_res_blocks  # 2
        self.resolution = resolution  # 128
        self.in_channels = in_channels  # 2
        self.sar_channels = in_ch
        # self.device = config.device

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        self.conv_in = torch.nn.Conv2d(self.in_channels,  # 2
                                       self.ch,  # 128
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)
        # self.conv_in_fft = torch.nn.Conv2d(4,  # 2
        #                                self.ch,  # 128
        #                                kernel_size=3,
        #                                stride=1,
        #                                padding=1)
        self.complexconv_in = ComplexConv2d(1,  # 2
                                       self.ch,  # 128
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        # self.conv_in2 = torch.nn.Conv2d(self.sar_channels,  # 2
                                        # self.ch,  # 128
                                        # kernel_size=3,
                                        # stride=1,
                                        # padding=1)
        # conv = layers.ComplexConv
        # diff = layers.DivLayer
        # prototype_size = 128
        # outsize = 4
        # self.wfm1 = conv(1, 16, kern_size=3, stride=(
        #     2, 2), reflect=1, new_init=True, use_groups_init=True, bias=False)
        #
        # self.wfm2 = conv(16, 32, kern_size=3,
        #                  stride=(2, 2), reflect=1, groups=2, new_init=True, use_groups_init=True)
        # self.wfm3 = conv(32, 64, kern_size=3,
        #                  stride=(2, 2), reflect=1, groups=4, new_init=True, use_groups_init=True)
        # self.wfm4 = conv(64, 64, kern_size=4, groups=64,
        #                  new_init=True, use_groups_init=True)
        #
        # self.diff1 = diff(16, 3, reflect=1, new_init=True)
        #
        # self.gtrelu1 = layers.GTReLU(16, phase_scale=True)
        # self.gtrelu2 = layers.GTReLU(32, phase_scale=True)
        # self.gtrelu3 = layers.GTReLU(64, phase_scale=True)
        #
        # self.fc1 = conv(64, prototype_size, 1, groups=4, new_init=True)
        #
        # self.bn = nn.BatchNorm1d(prototype_size * 2)
        #
        # dist_feat = layers.DistFeatures
        #
        # self.dist_feat = dist_feat(prototype_size, outsize)

        curr_res = resolution  # 128
        groups = 1
        diff = True
        in_ch_mult = (1,) + ch_mult  # (1, 1, 1, 2, 3, 4)
        self.down = nn.ModuleList()
        self.down_complex = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):  # 5 (0,1,2,3,4)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]  # 128、128、128、256、384
            block_out = ch * ch_mult[i_level]  # 128、128、256、384、512
            for i_block in range(self.num_res_blocks):  # 2
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:  # [32,16,8]
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions - 1:  # 4
                down.downsample = Downsample(block_in, resamp_with_conv)
                curr_res = curr_res // 2  # 64、32、16、8
            self.down.append(down)
        # complex data
        for i_level in range(self.num_resolutions):  # 5 (0,1,2,3,4)
            block_complex = nn.ModuleList()
            # attn_complex = nn.ModuleList()
            block_in = ch * in_ch_mult[i_level]  # 128、128、128、256、384
            block_out = ch * ch_mult[i_level]  # 128、128、256、384、512
            for i_block in range(self.num_res_blocks):  # 2
                block_complex.append(ResnetBlock_complex(in_channels=block_in,
                                                         out_channels=block_out,
                                                         temb_channels=self.temb_ch,
                                                         dropout=dropout))
                block_in = block_out
                # if curr_res in attn_resolutions: # [32,16,8]
                # attn_complex.append(AttnBlock_complex(block_in))
            down_complex = nn.Module()
            down_complex.block_complex = block_complex
            # down_complex.attn_complex = attn_complex
            if i_level != self.num_resolutions - 1:  # 4
                down_complex.downsample = Downsample_complex(block_in, resamp_with_conv, groups=groups, diff=diff)
                groups *= 2
                diff = False
                curr_res = curr_res // 2  # 64、32、16、8
            self.down_complex.append(down_complex)

        # 通道数和空间数都没有改变
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       temb_channels=self.temb_ch,
                                       dropout=dropout)

        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):  # 5 (4,3,2,1,0)
            block = nn.ModuleList()
            attn = nn.ModuleList()
            # mdaf = nn.ModuleList()
            block_out = ch * ch_mult[i_level]  # 512,384,256,128,128
            skip_in = ch * ch_mult[i_level]  # 512,384,256,128,128
            for i_block in range(self.num_res_blocks + 1):  # 3 (0,1,2)
                if i_block == self.num_res_blocks:  # 2
                    skip_in = ch * in_ch_mult[i_level]  # 384、256、128、128、128
                # mdaf.append(MDAF(dim=skip_in))
                block.append(ResnetBlock(in_channels=block_in + skip_in,
                                         out_channels=block_out,
                                         temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:  # [8,16,32]
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            # up.mdaf = mdaf
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2  # [16,32,64,128]
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        # self.conv_out = torch.nn.Conv2d(block_in,out_ch,kernel_size=3,stride=1,padding=1)
        self.conv_out = zero_module(torch.nn.Conv2d(block_in, out_ch, kernel_size=3, stride=1, padding=1))

    def upsample_concate(self, hs_fft):
        hs_fft_real = torch.real(torch.fft.ifftn(hs_fft, dim=(-2, -1)))
        return hs_fft_real

    def ComplextoMod(self, x):
        complex_value = torch.complex(x[:, 0, :, :, :], x[:, 1, :, :, :]).to(torch.complex64)
        # mod_value = torch.abs(complex_value)
        return complex_value

    def forward(self, x, t):
        assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        temb = get_timestep_embedding(t, self.ch)
        temb = self.temb.dense[0](temb)
        temb = nonlinearity(temb)
        temb = self.temb.dense[1](temb)  # temb[72,512]

        # temb=t
        # Complex SAR
        # mod_x = utils.complex2mod(x[:, :4, :, :]) # (1,2,256,256)
        # mod_x = torch.cat((mod_x, x[:, 4:, :, :]), dim=1)
        # complex_x = utils.toComplex(x[:, :4, :, :]) # (1,2,256,256)
        # complex_x = torch.fft.fftn(complex_x, dim=(-2, -1))
        # fft_xt = torch.fft.fftn(x[:, 4:, :, :], dim=(-2, -1))
        # complex_x = torch.cat((complex_x, fft_xt), dim=1).to(torch.complex64)
        # C2Matrix
        mod_x = utils.singleComplex2mod(x[:, 1:3, :, :])  # (64,1,64,64)
        mod_x = torch.cat((x[:, :1, :, :], mod_x), dim=1)
        mod_x = torch.cat((mod_x, x[:, 3:, :, :]), dim=1)
        complex_x = utils.singleToComplex(x[:, 1:3, :, :])  # (64,1,64,64)
        # fft_xt = torch.fft.fftn(x[:, 4:, :, :], dim=(-2, -1))
        # complex_x = torch.cat((complex_x, fft_xt), dim=1).to(torch.complex64)

        # downsampling
        xx = self.conv_in(mod_x)
        complex_xx = self.complexconv_in(complex_x)
        # debug

        # down1 = self.wfm1(complex_x) # torch.Size([64, 2, 16, 32, 32])
        # hs_fft_cds.append(self.ComplextoMod(down1)) # torch.Size([64, 16, 32, 32])
        # diff1 = self.diff1(down1) # torch.Size([64, 2, 16, 32, 32])
        # relu1 = self.gtrelu1(diff1) # torch.Size([64, 2, 16, 32, 32])
        # down2 = self.wfm2(relu1) # torch.Size([64, 2, 32, 16, 16])
        # relu2 = self.gtrelu2(down2) # torch.Size([64, 2, 32, 16, 16])
        # down3 = self.wfm3(relu2) # torch.Size([64, 2, 64, 8, 8])
        # relu3 = self.gtrelu3(down3) # torch.Size([64, 2, 64, 8, 8])
        # down4 = self.wfm4(relu3) # torch.Size([64, 2, 64, 5, 5])
        #
        # final = self.fc1(down4) # torch.Size([64, 2, 128, 5, 5])
        # x_shape = final.shape
        # final_bn = self.bn(x.reshape(x.shape[0], -1)).reshape(x_shape)

        hs = [xx]  # 先3×3卷积操作
        # hs_fft = [complex_xx]
        hs_fft_cds = [complex_xx]

        for i_level in range(self.num_resolutions):  # 5 (0,1,2,3,4)
            for i_block in range(self.num_res_blocks):  # 2 (0,1)
                h = self.down[i_level].block[i_block](hs[-1], temb)  # Resblock
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)  # attention
                hs.append(h)
            if i_level != self.num_resolutions - 1:
                hs.append(self.down[i_level].downsample(hs[-1]))
        # b-complex
        for i_level in range(self.num_resolutions):  # 5 (0,1,2,3,4)
            for i_block in range(self.num_res_blocks):  # 2 (0,1)
                h = self.down_complex[i_level].block_complex[i_block](hs_fft_cds[-1])  ##, temb) #temb_fft Resblock
                # print('1',h.shape)
                hs_fft_cds.append(h)
            if i_level != self.num_resolutions - 1:
                temp = torch.stack([hs_fft_cds[-1].real, hs_fft_cds[-1].imag], dim=1)
                # print('2',self.ComplextoMod(self.down_complex[i_level].downsample(hs_fft_cds[-1])).shape)
                hs_fft_cds.append(self.ComplextoMod(self.down_complex[i_level].downsample(temp)))


        # print(hs_fft[-1].shape, hs_fft[-1].dtype)
        # v3 ifftn转换
        # hs_fft_C = torch.real(torch.fft.ifftn(hs_fft[-1], dim=(-2, -1)))  # 逆变换之后虚部为0
        # hs_fft_C = torch.cat((torch.abs(hs_fft[-1][:, :2, :, :]), torch.real(hs_fft_C)), dim=1)
        # hs_fft_C = torch.real(hs_fft[-1])  # 逆变换之后虚部为0
        #
        # hs_fft_C = torch.real(hs_fft_C)  # 只保留实部
        # middle
        # print(hs[-1].dtype)
        # print(hs_fft_C.dtype)
        h = hs[-1] + torch.abs(hs_fft_cds[-1])  ###C +
        h = self.mid.block_1(h, temb)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h, temb)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):
            for i_block in range(self.num_res_blocks + 1):
                h = self.up[i_level].block[i_block](
                    # v3 ifftn转换
                    torch.cat([h, (hs.pop() + torch.abs(hs_fft_cds.pop()))], dim=1),
                    temb)  # pop删除并返回最后一个元素
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h0 = self.conv_out(h)
        return h0

