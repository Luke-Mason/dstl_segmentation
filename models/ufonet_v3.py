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
        depth_conv = nn.Conv2d(ninput, ninput, kernel_size=3, padding=1, groups=ninput)
        point_conv = nn.Conv2d(ninput, noutput - ninput, kernel_size=1, stride=2, padding=1, bias=True)

        self.depthwise_separable_conv = nn.Sequential(depth_conv, point_conv)
        # self.conv = nn.Conv2d(ninput, noutput-ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        x1 = self.pool(input)
        x2 = self.depthwise_separable_conv(input)

        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])

        output = torch.cat([x2, x1], 1)
        output = self.bn(output)
        output = self.relu(output)
        return output


def diff_pad(x1, x2):
    diffY = x2.size()[2] - x1.size()[2]
    diffX = x2.size()[3] - x1.size()[3]

    x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                    diffY // 2, diffY - diffY // 2])
    return x1


class UpsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)

        return F.relu(output)


class CDilated(nn.Module):
    '''
    This class defines the dilated convolution, which can maintain feature map size
    '''

    def __init__(self, nIn, nOut, kSize=3, stride=1, d=1):
        '''
        :param nIn: number of input channels
        :param nOut: number of output channels
        :param kSize: kernel size
        :param stride: optional stride rate for down-sampling
        :param d: optional dilation rate
        '''
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv2d(nIn, nOut, (kSize, kSize), stride=stride, padding=(padding, padding), bias=False,
                              dilation=d)

    def forward(self, input):
        '''
        :param input: input feature map
        :return: transformed feature map
        '''
        output = self.conv(input)
        return output


class InputProjectionA(nn.Module):
    '''
    This class projects the input image to the same spatial dimensions as the feature map.
    For example, if the input image is 512 x512 x3 and spatial dimensions of feature map size are 56x56xF, then
    this class will generate an output of 56x56x3, for input reinforcement, which establishes a direct link between
    the input image and encoding stage, improving the flow of information.
    '''

    def __init__(self, samplingTimes):
        '''
        :param samplingTimes: The rate at which you want to down-sample the image
        '''
        super().__init__()
        self.pool = nn.ModuleList()
        for i in range(0, samplingTimes):
            # pyramid-based approach for down-sampling
            self.pool.append(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, input):
        '''
        :param input: Input RGB Image
        :return: down-sampled image (pyramid-based approach)
        '''
        for pool in self.pool:
            input = pool(input)
        return input


class UFONet(BaseModel):
    def __init__(self, num_classes, **_):
        super().__init__()

        # self.layers = nn.ModuleList()

        self.down1 = DownsamplerBlock(3, 64 - 16)
        self.down2 = DownsamplerBlock(64, 128 - 32)
        self.down3 = DownsamplerBlock(128, 256 - 64)

        self.dilate1 = CDilated(nIn=3, nOut=16, kSize=3, d=8)
        self.dilate2 = CDilated(nIn=3, nOut=32, kSize=3, d=4)
        self.dilate3 = CDilated(nIn=3, nOut=64, kSize=3, d=2)

        self.sample1 = InputProjectionA(1)
        self.sample2 = InputProjectionA(2)
        self.sample3 = InputProjectionA(3)

        self.up1 = UpsamplerBlock(256, 64)
        self.up2 = UpsamplerBlock(64, 32)

        # self.up_dilate1 = CDilated(nIn=64, nOut=64, kSize=3, d=4)
        # self.up_dilate2 = CDilated(nIn=32, nOut=32, kSize=3, d=8)
        self.depth_conv1 = nn.Conv2d(64, 64, kernel_size=3, padding=1, groups=64)
        self.depth_conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1, groups=32)

        self.output_conv = nn.ConvTranspose2d(32, num_classes, 2, stride=2, padding=0, output_padding=0, bias=True)

    def forward(self, input):
        output = self.down1(input)
        sample = self.sample1(input)
        dilate1_sample = self.dilate1(sample)
        output = diff_pad(output, dilate1_sample)
        output = torch.cat([dilate1_sample, output], 1)  # 64
        # output = dilate1_sample + output

        output = self.down2(output)
        sample = self.sample2(input)
        dilate2_sample = self.dilate2(sample)
        output = diff_pad(output, dilate2_sample)
        output = torch.cat([dilate2_sample, output], 1)
        # output = dilate2_sample + output

        output = self.down3(output)
        sample = self.sample3(input)
        dilate3_sample = self.dilate3(sample)
        output = diff_pad(output, dilate3_sample)
        output = torch.cat([dilate3_sample, output], 1)
        # output = dilate3_sample + output

        output = self.up1(output)
        output = self.depth_conv1(output)
        output = self.up2(output)
        output = self.depth_conv2(output)

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
