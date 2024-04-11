import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
from torch import nn


class luminance_feedback(nn.Module):
    def __init__(self, init_weights=True):
        super(luminance_feedback, self).__init__()
        self.conv1 = reflect_conv_bn(in_channels=3, out_channels=16, kernel_size=7,stride=1,pad=3)
        self.conv2 = reflect_conv_bn(in_channels=3, out_channels=16, kernel_size=5,stride=1,pad=2)
        self.conv3 = reflect_conv_bn(in_channels=3, out_channels=16, kernel_size=3,stride=1,pad=1)
        self.conv4 = reflect_conv_bn(in_channels=48, out_channels=64, kernel_size=3,stride=1,pad=1)
        self.conv5 = reflect_conv(in_channels=64, out_channels=128, kernel_size=3,stride=1,pad=1)
        self.linear1 = nn.Linear(in_features=128, out_features=128)
        self.linear2 = nn.Linear(in_features=128, out_features=2)
        if init_weights:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        activate = nn.LeakyReLU(inplace=True)
        x1=activate(self.conv1(x))#1
        x2=activate(self.conv2(x))
        x3=activate(self.conv3(x))
        x=torch.concat([x1,x2,x3],dim=1)
        x=activate(self.conv4(x))
        x=activate(self.conv5(x))

        x = nn.AdaptiveAvgPool2d(1)(x) # GAP，先自适应池化为(b,c,1,1)->(b,c*1*1)
        x = x.view(x.size(0), -1) # 类似于tf.flatten()
        x = self.linear1(x) # FC层
        x = activate(x)
        x = self.linear2(x)
        x = nn.ReLU()(x)  # 设置ReLU激活函数，过滤负值
        return x
    
class reflect_conv_bn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad):
        super(reflect_conv_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=pad),
            nn.BatchNorm2d(out_channels))
    def forward(self, x):
        out = self.conv(x)
        return out
    
class reflect_conv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, pad):
        super(reflect_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=pad))

    def forward(self, x):
        out = self.conv(x)
        return out
