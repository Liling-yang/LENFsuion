import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import dataloader
import numpy as np
from torchstat import stat
from dual_attention_fusion_module import attention_fusion_weight
from dataloader import rgb2ycbcr,ycbcr2rgb

# img.shape=(b,c,h,w),input=vi(RGB)
class luminance_adjustment(nn.Module):
    def __init__(self):
        super(luminance_adjustment, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        number_f = 32
        self.e_conv1 = nn.Conv2d(3,number_f,3,1,1,bias=True) 
        self.e_conv2 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv3 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv4 = nn.Conv2d(number_f,number_f,3,1,1,bias=True) 
        self.e_conv5 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv6 = nn.Conv2d(number_f*2,number_f,3,1,1,bias=True) 
        self.e_conv7 = nn.Conv2d(number_f*2,24,3,1,1,bias=True) 

    def forward(self, x):
        x1 = self.relu(self.e_conv1(x))
        x2 = self.relu(self.e_conv2(x1))
        x3 = self.relu(self.e_conv3(x2))
        x4 = self.relu(self.e_conv4(x3))
        x5 = self.relu(self.e_conv5(torch.cat([x3,x4],1)))
        x6 = self.relu(self.e_conv6(torch.cat([x2,x5],1)))

        x_r = F.tanh(self.e_conv7(torch.cat([x1,x6],1)))
        r1,r2,r3,r4,r5,r6,r7,r8 = torch.split(x_r, 3, dim=1)

        x = x + r1*((torch.pow(x,2)-x)/torch.exp(x))
        x = x + r2*((torch.pow(x,2)-x)/torch.exp(x))
        x = x + r3*((torch.pow(x,2)-x)/torch.exp(x))
        enhance_image_1 = x + r4*((torch.pow(x,2)-x)/torch.exp(x))
        x = enhance_image_1 + r5*((torch.pow(enhance_image_1,2)-enhance_image_1)/torch.exp(enhance_image_1))
        x = x + r6*((torch.pow(x,2)-x)/torch.exp(x))
        x = x + r7*((torch.pow(x,2)-x)/torch.exp(x))
        enhance_image = x + r8*((torch.pow(x,2)-x)/torch.exp(x))
        r = torch.cat([r1,r2,r3,r4,r5,r6,r7,r8],1)
        return enhance_image_1,enhance_image,r
        

class ConvLeakyRelu2d(nn.Module):
    # convolution + leaky relu
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1,bias=True):
        super(ConvLeakyRelu2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding,stride=stride,bias=bias)
        self.bn = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        return F.leaky_relu(self.bn(self.conv1(x)), negative_slope=0.2)

class get_layer1(nn.Module):
    def __init__(self, num_channels, growth):
        super(get_layer1, self).__init__()
        self.conv_1 = ConvLeakyRelu2d(in_channels=num_channels, out_channels=32, kernel_size=3, stride=1, padding=1,bias=True)
        self.conv_2 = ConvLeakyRelu2d(in_channels=32, out_channels=growth, kernel_size=3, stride=1, padding=1,bias=True)
    def forward(self, x):
        x1 = self.conv_1(x)
        x1 = self.conv_2(x1)
        return x1

class denselayer(nn.Module):
    def __init__(self, num_channels, growth):
        super(denselayer, self).__init__()
        self.conv_1 = ConvLeakyRelu2d(in_channels=num_channels, out_channels=growth, kernel_size=3, stride=1, padding=1,bias=True)
        self.sobel = juanji_sobelxy(num_channels)
        self.sobel_conv = nn.Conv2d(num_channels, growth, 1, 1, 0)
    def forward(self, x):
        x1 = self.conv_1(x)
        x2 = self.sobel(x)
        x2 = self.sobel_conv(x2)
        return F.leaky_relu(x1+x2,negative_slope=0.1)

class encoder(nn.Module):
    def __init__(self):
        super(encoder, self).__init__()
        self.num_channels = 2
        self.num_features = 64
        self.growth = 64
        self.conv_layer1 = get_layer1(self.num_channels,self.num_features)
        self.conv_layer2 = denselayer(self.num_features,self.growth)
        self.conv_layer3 = denselayer(self.num_features*2,self.growth)
        self.conv_layer4 = denselayer(self.num_features*3,self.growth)
    def forward(self, x):
        x_max = torch.max(x, dim=1, keepdim=True).values
        x = torch.cat([x_max, x],dim=1)
        layer1 = self.conv_layer1(x)#in 2,out 64
        layer2 = self.conv_layer2(layer1)#in 64,out 64
        layer2 = torch.cat([layer2,layer1],dim=1)
        layer3 = torch.cat([layer2,self.conv_layer3(layer2)],dim=1)#in 64,out 64*3
        layer4 = torch.cat([layer3,self.conv_layer4(layer3)],dim=1)#in 64*3,out 64*4
        return layer4

# input=feature_y_f(经过fusion_net后的融合图像),output=y_f
class decoder(nn.Module):
    def __init__(self):
        super(decoder, self).__init__()
        self.Lrelu = nn.LeakyReLU()
        filter_n = 32
        self.de_conv1 = nn.Conv2d(filter_n*8,filter_n*4,3,1,1,bias=True)
        self.de_conv2 = nn.Conv2d(filter_n*4,filter_n*2,3,1,1,bias=True)
        self.de_conv3 = nn.Conv2d(filter_n*2,filter_n,3,1,1,bias=True)
        self.de_conv4 = nn.Conv2d(filter_n,1,3,1,1,bias=True)
        self.bn1 = nn.BatchNorm2d(filter_n*4)
        self.bn2=nn.BatchNorm2d(filter_n*2)
        self.bn3=nn.BatchNorm2d(filter_n)
        self.rgb2ycbcr = dataloader.rgb2ycbcr
        self.ycbcr2rgb = dataloader.ycbcr2rgb
    def forward(self,feature):
        feature=self.Lrelu(self.bn1(self.de_conv1(feature)))
        feature=self.Lrelu(self.bn2(self.de_conv2(feature)))
        feature=self.Lrelu(self.bn3(self.de_conv3(feature)))
        Y_f=torch.tanh(self.de_conv4(feature))
        return Y_f

# input=feature_vi,feature_ir(都是单通道的),output=feature_y_f
class FusionNet(nn.Module):
    def __init__(self):
        super(FusionNet,self).__init__()
        self.encoder=encoder().cuda()
        self.decoder=decoder().cuda()
    def forward(self,vi_clahe_y,ir):#vi_en其实input的是vi_en_ycbcr
        ir_orig = ir
        feature_vi_en = self.encoder(vi_clahe_y)
        feature_ir = self.encoder(ir_orig)
        feature_y_f = attention_fusion_weight(feature_vi_en, feature_ir)
        Y_f = self.decoder(feature_y_f)

        save_ir=self.decoder(feature_ir)
        save_vi_en=self.decoder(feature_vi_en)
        save_y_f=self.decoder(feature_y_f)

        return save_ir,save_vi_en,save_y_f,Y_f

class juanji_sobelxy(nn.Module):
    def __init__(self,channels, kernel_size=3, padding=1, stride=1, dilation=1, groups=1):
        super(juanji_sobelxy, self).__init__()
        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])
        self.convx=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convx.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.convy=nn.Conv2d(channels, channels, kernel_size=kernel_size, padding=padding, stride=stride, dilation=dilation, groups=channels,bias=False)
        self.convy.weight.data.copy_(torch.from_numpy(sobel_filter.T))
    def forward(self, x):
        sobelx = self.convx(x)
        sobely = self.convy(x)
        x=torch.abs(sobelx) + torch.abs(sobely)
        return x

class BN_Conv2d(nn.Module):
    def __init__(self, in_channels):
        super(BN_Conv2d, self).__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_channels,64,3,1,1,bias=True),
            nn.BatchNorm2d(64))
        self.Lrelu = nn.LeakyReLU()
    def forward(self, x):
        out=self.Lrelu(self.seq(x))
        out=out.cuda()
        return out
    