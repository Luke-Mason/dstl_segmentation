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

        self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        #self.conv3x1 = nn.Conv2d(noutput, noutput, (3,1), stride=1, padding=(2,0), bias=True, dilation = (2,1))
        #self.conv1x3 = nn.Conv2d(noutput, noutput, (1,3), stride=1, padding=(0,2), bias=True, dilation = (1,2))
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

class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):

        output = self.conv(input)
        output = self.bn(output)

        return F.relu(output)

class UFONet(BaseModel):
    def __init__(self, num_classes, **_):
        super().__init__()

        self.initial_block = DownsamplerBlock(3,16)

        self.layers = nn.ModuleList()

        self.layers.append(DownsamplerBlock(16,64))

        self.layers.append(DownsamplerBlock(64,128))

        self.layers.append(UpsamplerBlock(128, 64))

        self.layers.append(UpsamplerBlock(64,16))

        self.output_conv = nn.ConvTranspose2d( 16, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

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

    # """print layers and params of network"""
    # if __name__ == '__main__':
    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     model = UFONet(19).to(device)
    #     summary(model, (3, 512, 1024))
    #     # print(model)