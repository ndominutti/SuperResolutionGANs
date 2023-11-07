import torch
import torch.nn as nn
import torch.nn.functional as F

class Prework(nn.Module):
    def __init__(self,
                n_channels, 
                out_channels, 
                kernel_size, 
                stride, 
                padding,
                generator=True
                ):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding, bias=True)
        self.actfunc = nn.PReLU(num_parameters=out_channels) if generator else nn.LeakyReLU(.2, inplace=True) 

    def forward(self, x):
        return self.actfunc(self.conv(x))

class ConvBlockBase(nn.Module):
    def __init__(self, 
                n_channels, 
                out_channels, 
                kernel_size, 
                stride,
                padding,
                apply_bn=True,
                apply_act=True,
                generator=True
                ):
        super().__init__()
        self.apply_act = apply_act
        self.conv  = nn.Conv2d(n_channels, out_channels, kernel_size, stride, padding)
        self.bn    = nn.BatchNorm2d(out_channels) if apply_bn else nn.Identity() 
        self.prelu = nn.PReLU(num_parameters=out_channels) if generator else nn.LeakyReLU(.2, inplace=True)

    def forward(self, x):
        return self.prelu(self.bn(self.conv(x))) if self.apply_act else self.bn(self.conv(x))

class ResBlock(nn.Module):
    def __init__(self,
                channels,
                kernel_size=3,
                stride=1,
                padding=1):
        super().__init__()
        self.resblock1 = ConvBlockBase(channels, channels, kernel_size, stride, padding)
        self.resblock2 = ConvBlockBase(channels, channels, kernel_size, stride, padding, 
                                       apply_act=False)

    def forward(self, x):
        x_rb1 = self.resblock1(x)
        return self.resblock2(x_rb1)+x

class Upsampler(nn.Module):
    def __init__(self,
                n_channels,
                upsampling_factor=2,
                kernel_size=3,
                stride=1,
                padding=1
                ):
        super().__init__()
        self.conv = nn.Conv2d(n_channels, n_channels * upsampling_factor ** 2, kernel_size, stride, padding)
        self.pixelshuffle = nn.PixelShuffle(upsampling_factor) #n_channels*4, H, W -> n_channels, 2*H, 2*W
        self.prelu = nn.PReLU(num_parameters=n_channels)

    def forward(self,x):
        a = self.conv(x)
        return self.prelu(self.pixelshuffle(a))


# -------- ACTUAL MODELS -------- #

class Generator(nn.Module):
    def __init__(self,
                 in_channels=3,
                 out_channels=64,
                 resblocks=16,
                 upsamplers=2):
        super().__init__()
        self.prework         = Prework(in_channels, out_channels, 9, 1, 4)
        self.residual_blocks = nn.Sequential(*[ResBlock(out_channels) for _ in range(resblocks)])
        self.skipblock       = ConvBlockBase(
            out_channels, 
            out_channels, 
            3, 
            1,
            1,
            apply_act=False
        )
        self.upsampler       = nn.Sequential(*[Upsampler(out_channels) for _ in range(upsamplers)])
        self.conv            = nn.Conv2d(64, 3, kernel_size=9, stride=1, padding=4)

    def forward(self, x):
        x = self.prework(x)
        x_resblocks = self.residual_blocks(x)
        x = self.skipblock(x_resblocks) + x
        x = self.upsampler(x)
        return torch.tanh(self.conv(x)) #not really clear in the paper, this is an assumption

class Discriminator(nn.Module):
    def __init__(self,
                 n_channels=3,
                 feature_maps=[64,64,128,128,256,256,512,512]):
        super().__init__()
        self.prework   = Prework(
            n_channels, 
            feature_maps[0], 
            3, 
            1, 
            1,
            generator=False
        )
        self.disblocks = nn.Sequential(*[ConvBlockBase(
            feature_maps[i-1], 
            feature_maps[i], 
            3, 
            stride= 1 + i%2, #2, 1, 2, 1, ...
            padding=1,
            generator=False
        ) for i in range(1, len(feature_maps))])
        self.flatten     = nn.Flatten()
        self.dense1024 = nn.Linear(feature_maps[-1]*6*6, 1024) #Fixed for 24x24 input for now
        self.lrelu     = nn.LeakyReLU(.2, inplace=True)
        self.dense1    = nn.Linear(1024, 1)
        #self.sigmoid   = nn.Sigmoid()

    def forward(self, x):
        x = self.prework(x)
        x = self.disblocks(x)
        x = self.dense1024(self.flatten(x))
        x = self.lrelu(x)
        x = self.dense1(x)
        return x #self.sigmoid(x) -- Do not apply sigmoid due to BCEWithLogitsLoss usage for training as it's more numerically stable than using a plain Sigmoid followed by a BCELoss