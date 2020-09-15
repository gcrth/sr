import functools
import torch
import torch.nn as nn
import torch.nn.functional as F
import models.archs.arch_util as arch_util

from models.archs.arch_util import Separable_Conv2d


class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64,  bias=True):
        super(ResidualDenseBlock, self).__init__()
        self.dwconv1 = nn.Conv2d(in_channels=nf // 2, out_channels=nf // 2,
                                 kernel_size=3, stride=1, padding=1, dilation=1, groups=nf // 2, bias=True)
        self.pwconv1 = nn.Conv2d(in_channels=nf, out_channels=nf, kernel_size=1,
                                 stride=1, padding=0, dilation=1, groups=1, bias=True)
        self.dwconv2 = nn.Conv2d(in_channels=nf * 2, out_channels=nf * 2, kernel_size=3,
                                 stride=1, padding=1, dilation=1, groups=nf * 2, bias=True)
        self.pwconv2 = nn.Conv2d(in_channels=nf * 4, out_channels=nf // 2,
                                 kernel_size=1, stride=1, padding=0, dilation=1, groups=1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # initialization
        arch_util.initialize_weights(
            [self.dwconv1, self.pwconv1, self.dwconv2, self.pwconv2], 0.1)

    def forward(self, x):
        x1, x2_0 = x.chunk(2, dim=1)

        x2_1 = self.dwconv1(x2_0)
        x2_2 = self.lrelu(self.pwconv1(torch.cat((x2_0, x2_1), 1)))
        x2_3 = self.lrelu(self.dwconv2(torch.cat((x2_0, x2_1, x2_2), 1)))
        x2_4 = self.lrelu(self.pwconv2(torch.cat((x2_0, x2_1, x2_2, x2_3), 1)))

        x = torch.cat((x1, x2_4), 1)
        x = arch_util.channel_shuffle(x, x.shape[1])

        return x


class RRDB(nn.Module):
    '''Residual in Residual Dense Block'''

    def __init__(self, nf):
        super(RRDB, self).__init__()
        self.RDB1 = ResidualDenseBlock(nf,  bias=True)
        self.RDB2 = ResidualDenseBlock(nf,  bias=True)
        self.RDB3 = ResidualDenseBlock(nf,  bias=True)

    def forward(self, x):
        out = self.RDB1(x)
        out = self.RDB2(out)
        out = self.RDB3(out)
        return out * 0.2 + x


class FastNet(nn.Module):
    def __init__(self, in_nc, out_nc, nf, nb):
        super(FastNet, self).__init__()
        BasicBlock_block_f = functools.partial(RRDB, nf=nf)

        self.conv_first = nn.Conv2d(in_nc, nf, 3, 1, 1, bias=True)
        self.RRDB_trunk = arch_util.make_layer(BasicBlock_block_f, nb)
        self.trunk_conv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        #### upsampling
        self.upconv1 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.upconv2 = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.HRconv = nn.Conv2d(nf, nf, 3, 1, 1, bias=True)
        self.conv_last = nn.Conv2d(nf, out_nc, 3, 1, 1, bias=True)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        fea = self.conv_first(x)
        trunk = self.trunk_conv(self.RRDB_trunk(fea))
        fea = fea + trunk

        fea = self.lrelu(self.upconv1(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        fea = self.lrelu(self.upconv2(F.interpolate(
            fea, scale_factor=2, mode='nearest')))
        out = self.conv_last(self.lrelu(self.HRconv(fea)))

        return out
