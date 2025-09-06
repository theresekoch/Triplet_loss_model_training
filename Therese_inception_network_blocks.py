import torch.nn as nn
import torch.nn.functional as F
import torch



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))


class InceptionBlock(nn.Module):
    def __init__( self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, red_7x7, out_7x7):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)

        self.branch2 = nn.Sequential(
                                     ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0), 
                                     ConvBlock(red_3x3, out_3x3, kernel_size = 3, padding = 1), )

        self.branch3 = nn.Sequential(
                                    ConvBlock(in_channels, red_5x5, kernel_size = 1), 
                                    ConvBlock(red_5x5, out_5x5, kernel_size = 5, padding = 2), )

        self.branch4 = nn.Sequential(
                                     ConvBlock(in_channels, red_7x7, kernel_size = 1, padding = 0), 
                                     ConvBlock(red_7x7, out_7x7, kernel_size = 7, padding = 3), )

        
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in  branches],  1)