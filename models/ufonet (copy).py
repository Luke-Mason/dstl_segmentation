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


class Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(Conv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding,
                               dilation=dilation,  bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

class DeConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, dilation=1, bias=False,
                 BatchNorm=nn.BatchNorm2d):
        super(DeConv2d, self).__init__()

        if dilation > kernel_size // 2:
            padding = dilation
        else:
            padding = kernel_size // 2

        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding=padding,
                               output_padding=1,dilation=dilation, bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        x = self.deconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UFONet(BaseModel):
    def __init__(self, num_classes, **_):
        super().__init__()
        self.pool = nn.MaxPool2d(2, stride=2)
        self.initial_block = Conv2d(3, 64)

        self.layers = nn.ModuleList()
        self.layers.append(Conv2d(64, 64))
        self.layers.append(self.pool)

        self.layers.append(Conv2d(64, 128))
        self.layers.append(Conv2d(128, 128))
        self.layers.append(self.pool)

        self.layers.append(Conv2d(128, 256))
        self.layers.append(Conv2d(256, 256))
        self.layers.append(self.pool)

        self.layers.append(Conv2d(256, 512))
        self.layers.append(Conv2d(512, 512))
        self.layers.append(self.pool)


        self.layers.append(nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, output_padding=0, bias=True))
        self.layers.append(Conv2d(256, 256))

        self.layers.append(nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, output_padding=0, bias=True))
        self.layers.append(Conv2d(128, 128))

        self.layers.append(nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, output_padding=0, bias=True))
        self.layers.append(Conv2d(64, 64))

        self.output_conv = nn.ConvTranspose2d(64, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

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
    model = UFONet(num_classes=2).to(device)
    summary(model, (3, 128, 128))
