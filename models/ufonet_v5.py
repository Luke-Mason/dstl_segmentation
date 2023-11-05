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

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv1 = nn.Conv2d(ninput, noutput, (3, 3), stride=1, padding=0, bias=True)
        self.conv2 = nn.Conv2d(noutput, noutput, (3, 3), stride=1, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2, padding=1)
        #self.conv3x1 = nn.Conv2d(noutput, noutput, (3,1), stride=1, padding=(2,0), bias=True, dilation = (2,1))
        #self.conv1x3 = nn.Conv2d(noutput, noutput, (1,3), stride=1, padding=(0,2), bias=True, dilation = (1,2))
        #self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv1(input)
        print((output.size()))
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.pool(output)

        return output

class CovBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput, (3, 3), stride=1, padding=1, bias=True)
        self.conv = nn.Conv2d(noutput, noutput, (3, 3), stride=1, padding=1, bias=True)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        output = self.conv(input)
        print(output.size())
        output = self.relu(output)
        output = self.conv(output)
        output = self.relu(output)

        return output

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv1 = nn.Conv2d(ninput, noutput, 3, stride=1, padding=1, bias=True)
        self.conv2 = nn.ConvTranspose2d(noutput, noutput, 2, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):

        output = self.conv1(input)
        output = self.conv2(output)
        #output = self.bn(output)

        return F.relu(output)

class UFONet(BaseModel):
    def __init__(self, num_classes, **_):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,64)
        self.relu = nn.ReLU(inplace=True)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(64,128))
        self.layers.append(DownsamplerBlock(128, 256))
        self.layers.append(DownsamplerBlock(256, 512))

        self.layers.append(CovBlock(512, 512))

        self.layers.append(UpsamplerBlock(512, 256))
        self.layers.append(UpsamplerBlock(256, 128))

        self.layers.append(UpsamplerBlock(128,64))

        self.output_conv = nn.ConvTranspose2d(64, num_classes, 2, stride=2, bias=True)

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
    model = UFONet(19).to(device)
    summary(model, (3, 512, 512))
    # print(model)