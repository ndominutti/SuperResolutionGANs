"""
This module implements the whole architecture from the Generator and the Discriminator NNs
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------  BUILDING BLOCKS -------- #


class Prework(nn.Module):
    def __init__(
        self, n_channels:int, out_channels:int, kernel_size:int, stride:int, padding:int, generator:bool=True
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            n_channels, out_channels, kernel_size, stride, padding, bias=True
        )
        self.actfunc = (
            nn.PReLU(num_parameters=out_channels)
            if generator
            else nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x:torch.Tensor):
        return self.actfunc(self.conv(x))


class ConvBlockBase(nn.Module):
    def __init__(
        self,
        n_channels:int,
        out_channels:int,
        kernel_size:int,
        stride:int,
        padding:int,
        apply_bn:bool=True,
        apply_act:bool=True,
        generator:bool=True,
    ):
        super().__init__()
        self.apply_act = apply_act
        self.conv = nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels) if apply_bn else nn.Identity()
        self.prelu = (
            nn.PReLU(num_parameters=out_channels)
            if generator
            else nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x:torch.Tensor):
        return (
            self.prelu(self.bn(self.conv(x)))
            if self.apply_act
            else self.bn(self.conv(x))
        )


class ResBlock(nn.Module):
    def __init__(self, channels:int, kernel_size:int=3, stride:int=1, padding:int=1):
        super().__init__()
        self.resblock1 = ConvBlockBase(channels, channels, kernel_size, stride, padding)
        self.resblock2 = ConvBlockBase(
            channels, channels, kernel_size, stride, padding, apply_act=False
        )

    def forward(self, x:torch.Tensor):
        x_rb1 = self.resblock1(x)
        return self.resblock2(x_rb1) + x


class Upsampler(nn.Module):
    def __init__(
        self, n_channels:int, upsampling_factor:int=2, kernel_size:int=3, stride:int=1, padding:int=1
    ):
        super().__init__()
        self.conv = nn.Conv2d(
            n_channels,
            n_channels * upsampling_factor**2,
            kernel_size,
            stride,
            padding,
        )
        self.pixelshuffle = nn.PixelShuffle(
            upsampling_factor
        )  # n_channels*4, H, W -> n_channels, 2*H, 2*W
        self.prelu = nn.PReLU(num_parameters=n_channels)

    def forward(self, x:torch.Tensor):
        a = self.conv(x)
        return self.prelu(self.pixelshuffle(a))


# --------  MODELS -------- #


class Generator(nn.Module):
    def __init__(self, in_channels:int=3, out_channels:int=64, resblocks:int=16, upsamplers:int=2):
        super().__init__()
        self.prework = Prework(in_channels, out_channels, 9, 1, 4)
        self.residual_blocks = nn.Sequential(
            *[ResBlock(out_channels) for _ in range(resblocks)]
        )
        self.skipblock = ConvBlockBase(
            out_channels, out_channels, 3, 1, 1, apply_act=False
        )
        self.upsampler = nn.Sequential(
            *[Upsampler(out_channels) for _ in range(upsamplers)]
        )
        self.conv = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x:torch.Tensor):
        x = self.prework(x)
        x_resblocks = self.residual_blocks(x)
        x = self.skipblock(x_resblocks) + x
        x = self.upsampler(x)
        return torch.tanh(
            self.conv(x)
        )  # not really clear in the paper, this is an assumption


class Discriminator(nn.Module):
    def __init__(
        self, n_channels:int=3, feature_maps:list=[64, 64, 128, 128, 256, 256, 512, 512]
    ):
        super().__init__()
        self.prework = Prework(n_channels, feature_maps[0], 3, 1, 1, generator=False)
        self.disblocks = nn.Sequential(
            *[
                ConvBlockBase(
                    feature_maps[i - 1],
                    feature_maps[i],
                    3,
                    stride=1 + i % 2,  # 2, 1, 2, 1, ...
                    padding=1,
                    generator=False,
                )
                for i in range(1, len(feature_maps))
            ]
        )
        self.flatten = nn.Flatten()
        self.dense1024 = nn.Linear(
            feature_maps[-1] * 6 * 6, 1024
        )  # Fixed for 24x24 input for now
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.dense1 = nn.Linear(1024, 1)
        # self.sigmoid   = nn.Sigmoid()

    def forward(self, x:torch.Tensor):
        x = self.prework(x)
        x = self.disblocks(x)
        x = self.dense1024(self.flatten(x))
        x = self.lrelu(x)
        x = self.dense1(x)
        return x  # self.sigmoid(x) -- Do not apply sigmoid due to BCEWithLogitsLoss usage for training as it's more numerically stable than using a plain Sigmoid followed by a BCELoss
