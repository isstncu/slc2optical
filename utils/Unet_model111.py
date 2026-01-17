import torch
import torch.nn as nn
import torch.nn.functional as F
from torchgan.layers import SpectralNorm2d
import enum
from torchsummary import summary
import numpy as np
from ssim import msssim
from normalization import SwitchNorm2d
import math
import einops
from einops import rearrange
import pywt

# This script is from the following repositories
# https://github.com/ermongroup/ddim
# https://github.com/bahjat-kawar/ddrm

NUM_BANDS = 4

class ReconstructionLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=1.0):
        super(ReconstructionLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, prediction, target):
        loss = (self.alpha * F.mse_loss(prediction, target) +
                self.gamma * (1.0 - torch.mean(F.cosine_similarity(prediction, target, 1))) +
                self.beta * (1.0 - msssim(prediction, target, normalize=True)))
        return loss

class Conv3X3NoPadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3NoPadding, self).__init__(in_channels, out_channels, 3, stride=stride, padding=1)


#3*3卷积模块 ReplicationPad2d(1)256变为258
class Conv3X3WithPadding(nn.Sequential):
    def __init__(self, in_channels, out_channels, stride=1):
        super(Conv3X3WithPadding, self).__init__(
            nn.ReplicationPad2d(1),#ReplicationPad2d:使用输入边界的复制来填充输入张量。大小变化：256*256变成258*258
            nn.Conv2d(in_channels, out_channels, 3, stride=stride)
        )


class LAConv2D(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, use_bias=True):
        super(LAConv2D, self).__init__()
        self.in_planes = in_planes
        self.out_planes = out_planes
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.bias = use_bias
        
        # Generating local adaptive weights
        self.attention1=nn.Sequential(
            nn.Conv2d(in_planes, kernel_size**2, kernel_size, stride, padding),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size**2,kernel_size**2,1),
            nn.ReLU(inplace=True),
            nn.Conv2d(kernel_size ** 2, kernel_size ** 2, 1),
            nn.Sigmoid()
        ) #b,9,H,W È«Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        # self.attention2=nn.Sequential(
        #     nn.Conv2d(in_planes,(kernel_size**2)*in_planes,kernel_size, stride, padding,groups=in_planes),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d((kernel_size**2)*in_planes,(kernel_size**2)*in_planes,1,groups=in_planes),
        #     nn.Sigmoid()
        # ) #b,9n,H,W µ¥Í¨µÀÏñËØ¼¶ºË×¢ÒâÁ¦
        if use_bias==True: # Global local adaptive weights
            self.attention3=nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_planes,out_planes,1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_planes,out_planes,1)
            ) #b,m,1,1 Í¨µÀÆ«ÖÃ×¢ÒâÁ¦

        conv1=nn.Conv2d(in_planes,out_planes,kernel_size,stride,padding,dilation,groups)
        self.weight=conv1.weight # m, n, k, k
    def forward(self,x):
        (b, n, H, W) = x.shape
        m=self.out_planes
        k=self.kernel_size
        n_H = 1 + int((H + 2 * self.padding - k) / self.stride)
        n_W = 1 + int((W + 2 * self.padding - k) / self.stride)
        atw1=self.attention1(x) #b,k*k,n_H,n_W
        #atw2=self.attention2(x) #b,n*k*k,n_H,n_W

        atw1=atw1.permute([0,2,3,1]) #b,n_H,n_W,k*k
        atw1=atw1.unsqueeze(3).repeat([1,1,1,n,1]) #b,n_H,n_W,n,k*k
        atw1=atw1.view(b,n_H,n_W,n*k*k) #b,n_H,n_W,n*k*k

        #atw2=atw2.permute([0,2,3,1]) #b,n_H,n_W,n*k*k

        atw=atw1#*atw2 #b,n_H,n_W,n*k*k
        atw=atw.view(b,n_H*n_W,n*k*k) #b,n_H*n_W,n*k*k
        atw=atw.permute([0,2,1]) #b,n*k*k,n_H*n_W

        kx=F.unfold(x,kernel_size=k,stride=self.stride,padding=self.padding) #b,n*k*k,n_H*n_W
        atx=atw*kx #b,n*k*k,n_H*n_W

        atx=atx.permute([0,2,1]) #b,n_H*n_W,n*k*k
        atx=atx.view(1,b*n_H*n_W,n*k*k) #1,b*n_H*n_W,n*k*k

        w=self.weight.view(m,n*k*k) #m,n*k*k
        w=w.permute([1,0]) #n*k*k,m
        y=torch.matmul(atx,w) #1,b*n_H*n_W,m
        y=y.view(b,n_H*n_W,m) #b,n_H*n_W,m
        if self.bias==True:
            bias=self.attention3(x) #b,m,1,1
            bias=bias.view(b,m).unsqueeze(1) #b,1,m
            bias=bias.repeat([1,n_H*n_W,1]) #b,n_H*n_W,m
            y=y+bias #b,n_H*n_W,m

        y=y.permute([0,2,1]) #b,m,n_H*n_W
        y=F.fold(y,output_size=(n_H,n_W),kernel_size=1) #b,m,n_H,n_W
        return y

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
    # swish
    return x*torch.sigmoid(x)


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class Upsample(nn.Module):
    def __init__(self, in_channels, with_conv):
        super().__init__()
        self.with_conv = with_conv
        self.UP = nn.Sequential(
            nn.Conv2d(in_channels, 4*in_channels, 3, stride=1, padding=1),
            nn.PixelShuffle(2)
            )
        if self.with_conv:
            self.conv = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        #x = torch.nn.functional.interpolate(x, scale_factor=2.0, mode="nearest")
        
        x = self.UP(x)
        if self.with_conv:
            x = self.conv(x)
        return x

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self):
        super(space_to_depth, self).__init__()
    def forward(self, x):
        return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

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

            self.DOWN = nn.Sequential(
                space_to_depth(),
                nn.Conv2d(4*in_channels, in_channels, 3, stride=1, padding=1)
                )

    def forward(self, x):
        if self.with_conv:
            pad = (0, 1, 0, 1)
            x = torch.nn.functional.pad(x, pad, mode="constant", value=0)
            x = self.conv(x)
            
            #x = self.DOWN(x)
        else:
            x = torch.nn.functional.avg_pool2d(x, kernel_size=2, stride=2)
        return x

# def zero_module(module):
    # """
    # Zero out the parameters of a module and return it.将模块的参数归零并返回
    # """
    # for p in module.parameters():
        # p.detach().zero_()
    # return module

class ResnetBlock(nn.Module):
    def __init__(self, *, in_channels, out_channels=None, conv_shortcut=False,
                 dropout, temb_channels=512):
        super().__init__()
        self.in_channels = in_channels
        out_channels = in_channels if out_channels is None else out_channels
        self.out_channels = out_channels
        self.use_conv_shortcut = conv_shortcut

        self.norm1 = Normalize(in_channels)
        self.conv1 = torch.nn.Conv2d(in_channels,out_channels,kernel_size=3,stride=1,padding=1)#torch.nn.Conv2d
        #self.temb_proj = torch.nn.Linear(temb_channels,out_channels)
        self.norm2 = Normalize(out_channels)
        self.dropout = torch.nn.Dropout(dropout)

        self.conv2 = torch.nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=1,padding=1)#torch.nn.Conv2d
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

    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = nonlinearity(h)
        h = self.conv1(h)

        h = h #+ self.temb_proj(nonlinearity(temb))[:, :, None, None]

        h = self.norm2(h)
        h = nonlinearity(h)
        h = self.dropout(h)
        h = self.conv2(h)

        if self.in_channels != self.out_channels:
            if self.use_conv_shortcut:
                x = self.conv_shortcut(x)
            else:
                x = self.nin_shortcut(x)

        return x+h


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
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x+h_

class Cross_AttnBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)#GN
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
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x1, x2):
        h1_ = x1
        h2_ = x2
        h1_ = self.norm(h1_)
        h2_ = self.norm(h2_)
        q = self.q(h1_)
        k = self.k(h2_)
        v = self.v(h2_)

        # compute attention
        b, c, h, w = q.shape
        q = q.reshape(b, c, h*w)
        q = q.permute(0, 2, 1)   # b,hw,c
        k = k.reshape(b, c, h*w)  # b,c,hw
        w_ = torch.bmm(q, k)     #计算两个tensor的矩阵乘法 # b,hw,hw    w[b,i,j]=sum_c q[b,i,c]k[b,c,j]
        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = v.reshape(b, c, h*w)
        w_ = w_.permute(0, 2, 1)   # b,hw,hw (first hw of k, second of q)
        # b, c,hw (hw of q) h_[b,c,j] = sum_i v[b,c,i] w_[b,i,j]
        h_ = torch.bmm(v, w_)
        h_ = h_.reshape(b, c, h, w)

        h_ = self.proj_out(h_)

        return x1+h_


class Gen(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device('cuda')
        ch, out_ch, ch_mult = 128, 4, tuple([1, 1, 2, 3, 4])
        num_res_blocks = 3
        attn_resolutions = [64, 16, ]
        dropout = 0.0
        in_channels = 4 * 2 #if config.data.conditional else config.model.in_channels
        resolution = 128
        resamp_with_conv = True

        self.ch = ch#128
        self.temb_ch = self.ch*4#128*4
        self.num_resolutions = len(ch_mult)#5
        self.num_res_blocks = num_res_blocks
        self.resolution = resolution
        self.in_channels = in_channels

        # timestep embedding
        self.temb = nn.Module()
        self.temb.dense = nn.ModuleList([
            torch.nn.Linear(self.ch,
                            self.temb_ch),
            torch.nn.Linear(self.temb_ch,
                            self.temb_ch),
        ])

        # downsampling
        self.conv_in = torch.nn.Conv2d(in_channels,
                                       self.ch,
                                       kernel_size=3,
                                       stride=1,
                                       padding=1)

        curr_res = resolution
        in_ch_mult = (1,)+ch_mult#(1,1,1,2,3,4)
        self.down = nn.ModuleList()
        block_in = None
        for i_level in range(self.num_resolutions):#5
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_in = ch*in_ch_mult[i_level]#128*(1,1,1,2,3,4)0:128-128,1:128-128,2:128-256,3:256-384,4:384-512
            block_out = ch*ch_mult[i_level]#128*(1,1,2,3,4)
            for i_block in range(self.num_res_blocks):
                block.append(ResnetBlock(in_channels=block_in,
                                         out_channels=block_out,
                                         #temb_channels=self.temb_ch,
                                         dropout=dropout))#128-128；128-256；256-384；384-512
                block_in = block_out
                if curr_res in attn_resolutions:#128,64,32, 16, 8     16
                    attn.append(AttnBlock(block_in))
            down = nn.Module()
            down.block = block
            down.attn = attn
            if i_level != self.num_resolutions-1:#4的时候0,1,2,3
                down.downsample = Downsample(block_in, resamp_with_conv)
                down.conv = torch.nn.Conv2d(1,block_in,kernel_size=1,stride=1,padding=0)
                down.CA = Cross_AttnBlock(block_in)
                curr_res = curr_res // 2#32；16；8
            self.down.append(down)

        # middle
        self.mid = nn.Module()
        self.mid.block_1 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       #temb_channels=self.temb_ch,
                                       dropout=dropout)
        self.mid.attn_1 = AttnBlock(block_in)
        self.mid.block_2 = ResnetBlock(in_channels=block_in,
                                       out_channels=block_in,
                                       #temb_channels=self.temb_ch,
                                       dropout=dropout)

        # upsampling
        self.up = nn.ModuleList()
        for i_level in reversed(range(self.num_resolutions)):#3,2,1,0
            block = nn.ModuleList()
            attn = nn.ModuleList()
            block_out = ch*ch_mult[i_level]
            skip_in = ch*ch_mult[i_level]#128*(1,2,3,4)
            for i_block in range(self.num_res_blocks+1):#0,1,2
                if i_block == self.num_res_blocks:#2
                    skip_in = ch*in_ch_mult[i_level]#128*(1,1,2,3,4)
                block.append(ResnetBlock(in_channels=block_in+skip_in,
                                         out_channels=block_out,
                                         #temb_channels=self.temb_ch,
                                         dropout=dropout))
                block_in = block_out
                if curr_res in attn_resolutions:#8,16 ,32,64
                    attn.append(AttnBlock(block_in))
            up = nn.Module()
            up.block = block
            up.attn = attn
            if i_level != 0:
                up.upsample = Upsample(block_in, resamp_with_conv)
                curr_res = curr_res * 2#8,16,32,64
            self.up.insert(0, up)  # prepend to get consistent order

        # end
        self.norm_out = Normalize(block_in)
        self.conv_out = torch.nn.Conv2d(block_in,
                                        out_ch,
                                        kernel_size=3,
                                        stride=1,
                                        padding=1)

    def forward(self, x):
        #assert x.shape[2] == x.shape[3] == self.resolution

        # timestep embedding
        #temb = get_timestep_embedding(t, self.ch)
        #temb = self.temb.dense[0](temb)
        #temb = nonlinearity(temb)
        #temb = self.temb.dense[1](temb)

        # downsampling
        im = x[0]
        img = im[:,0,:,:] + im[:,1,:,:] + im[:,2,:,:] + im[:,3,:,:]
        img = img.div_(4).cpu()
        img = np.expand_dims(img, axis=1)
        down_res = []
        [cl, (cH4, cV4, cD4), (cH3, cV3, cD3), (cH2, cV2, cD2), (cH1, cV1, cD1)] = pywt.wavedec2(img, 'db1', level=4)#8.16.32.64
        down_res.append(torch.from_numpy((np.absolute(cH1) + np.absolute(cV1)+ np.absolute(cD1))).to(self.device))#64
        down_res.append(torch.from_numpy((np.absolute(cH2) + np.absolute(cV2)+ np.absolute(cD2))).to(self.device))#32
        down_res.append(torch.from_numpy((np.absolute(cH3) + np.absolute(cV3)+ np.absolute(cD3))).to(self.device))#16
        down_res.append(torch.from_numpy((np.absolute(cH4) + np.absolute(cV4)+ np.absolute(cD4))).to(self.device))#8
        
        hs = [self.conv_in(torch.cat([x[0], x[1]],1))]
        h = hs[-1]
        for i_level in range(self.num_resolutions):#0,1,2,3,4
            for i_block in range(self.num_res_blocks):
                h = self.down[i_level].block[i_block](h)
                if len(self.down[i_level].attn) > 0:
                    h = self.down[i_level].attn[i_block](h)
                hs.append(h)
            if i_level != self.num_resolutions-1:#0,1,2
                h = self.down[i_level].downsample(h)
                out_d = self.down[i_level].conv(down_res[i_level])
                hs.append(self.down[i_level].CA(out_d,h))
        
        # middle
        h = hs[-1]
        h = self.mid.block_1(h)
        h = self.mid.attn_1(h)
        h = self.mid.block_2(h)

        # upsampling
        for i_level in reversed(range(self.num_resolutions)):#3,2,1,0
            for i_block in range(self.num_res_blocks+1):
                h = self.up[i_level].block[i_block](
                    torch.cat([h, hs.pop()], dim=1))#如果列表为空或索引超出范围，将引发IndexError。
                if len(self.up[i_level].attn) > 0:
                    h = self.up[i_level].attn[i_block](h)
            if i_level != 0:
                h = self.up[i_level].upsample(h)

        # end
        h = self.norm_out(h)
        h = nonlinearity(h)
        h = self.conv_out(h)
        return h


class D_ResidulBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        residual = [
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(Conv3X3NoPadding(in_channels, in_channels)),
            nn.MaxPool2d(2, stride=2),#MaxPool2d,AvgPool2d
            nn.BatchNorm2d(in_channels),
            nn.LeakyReLU(inplace=True),
            SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1))
        ]
        transform = [SpectralNorm2d(nn.Conv2d(in_channels, out_channels, 1)),
                     nn.MaxPool2d(2, stride=2)]

        self.residual = nn.Sequential(*residual)
        self.transform = nn.Sequential(*transform)

    def forward(self, inputs):

        trunk = self.residual(inputs)
        #print('trunk', trunk.shape)
        lateral = self.transform(inputs)
        #print('lateral', lateral.shape)
        return trunk + lateral

#鉴别器网络
class Dis(nn.Module):
    def __init__(self):
        super(Dis, self).__init__()
        channels = [32, 32, 64, 64, 128, 128, 256, 256]  # python中数组下标从0开始
        self.model = nn.Sequential(
            D_ResidulBlock(NUM_BANDS*2, channels[0]),#2-128
            D_ResidulBlock(channels[0], channels[1]),#4
            D_ResidulBlock(channels[1], channels[2]),#8
            D_ResidulBlock(channels[2], channels[3]),#16
            D_ResidulBlock(channels[3], channels[4]),#32
            D_ResidulBlock(channels[4], channels[5]),#64
            #D_ResidulBlock(channels[5], channels[6]),#128
            #D_ResidulBlock(channels[6], channels[7]),#256-1
            SpectralNorm2d(nn.Conv2d(channels[5], 1, 1))
        )
    def forward(self, inputs):
        out = self.model(inputs)
        #print('out',out.shape)
        return out

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Gen().to(device)
    #model = Dis().to(device)
    summary(model, input_size=(8, 64, 64))


if __name__ == '__main__':
    main()