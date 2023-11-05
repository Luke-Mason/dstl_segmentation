###################################################################################################
#ESNet: An Efficient Symmetric Network for Real-time Semantic Segmentation
#Paper-Link: https://arxiv.org/pdf/1906.09826.pdf
###################################################################################################

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torchsummary import summary
from base import BaseModel
from utils.helpers import initialize_weights
from itertools import chain


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class SeparableDeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(SeparableDeConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv1 = nn.ConvTranspose2d(in_channels, in_channels, kernel_size, stride, padding=padding,
                               output_padding=1,dilation=dilation, groups=in_channels, bias=bias)
        self.bn = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.ConvTranspose2d(in_channels, out_channels, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pointwise(x)
        return x

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = SeparableConv2d(ninput, noutput - ninput, 3, stride=2, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)

        output = self.bn(output)
        output = self.relu(output)
        return output


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = SeparableDeConv2d(ninput, noutput, 3, stride=2, bias=True)

        # depth_deconv = nn.ConvTranspose2d(ninput, ninput, kernel_size=3, padding=1,groups=ninput)
        # point_deconv = nn.ConvTranspose2d(ninput, noutput, kernel_size=1, stride=2, output_padding=1,
        #                                   bias=True)
        # self.conv = nn.Sequential(depth_deconv, point_deconv)

        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)

        return F.relu(output)

class UFONet(BaseModel):
    def __init__(self, num_classes, **_):
        super().__init__()

        self.initial_block = DownsamplerBlock(3, 16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16, 64))

        self.layers.append(DownsamplerBlock(64, 128))

        self.layers.append(SeparableConv2d(128, 128, kernel_size=3))

        #self.layers.append(nn.Dropout2d(0.3))

        self.layers.append(UpsamplerBlock(128, 64))

        self.layers.append(UpsamplerBlock(64, 16))

        self.output_conv = nn.ConvTranspose2d(16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.initial_block(input)

        for layer in self.layers:
            output = layer(output)

        output = self.output_conv(output)

        return output

    def get_backbone_params(self):
        # There is no backbone for unet, all the parameters are trained from scratch
        return []

    def get_decoder_params(self):
        return self.parameters()

    def freeze_bn(self):
        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d): module.eval()

"""print layers and params of network"""
if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UFONet(num_classes=19).to(device)
    summary(model, (3, 512, 512))
