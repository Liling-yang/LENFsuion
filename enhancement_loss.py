import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.vgg import vgg16

# input的x是待增强的RGB的vi
class L_color(nn.Module):
    def __init__(self):
        super(L_color, self).__init__()
    def forward(self, x):
        b,c,h,w = x.shape
        mean_rgb = torch.mean(x,[2,3],keepdim=True)
        mr,mg,mb = torch.split(mean_rgb, 1, dim=1)
        Drg = torch.abs(mr-mg)
        Drb = torch.abs(mr-mb)
        Dgb = torch.abs(mb-mg)
        k = torch.pow(torch.pow(Drg,2) + torch.pow(Drb,2) + torch.pow(Dgb,2),0.5)
        return k


# input是vi,vi_en(都是RGB的)
class L_spa(nn.Module):
    def __init__(self):
        super(L_spa, self).__init__()
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
        # 池化核尺寸维4*4，stride默认和kernel_size一样大小
        # 区域的size为4*4
    def forward(self, org , enhance ):
        b,c,h,w = org.shape
        # mean()函数的参数：dim=0，按列求平均值，返回的形状是（1，列数）
        # dim=1，按行求平均值，返回的形状是（行数，1）
        # 默认不设置dim时，返回所有元素的平均值
        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)
        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)

        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        return E

# input是vi(RGB)
class L_exp(nn.Module):
    def __init__(self,patch_size,mean_val):
        super(L_exp, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)
        self.mean_val = mean_val
    def forward(self, x ):
        b,c,h,w = x.shape
        x = torch.mean(x,1,keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean- torch.FloatTensor([self.mean_val]).cuda(),2))
        return d

# input是vi(RGB)，计算TV_LOSS，没改动
class L_TV(nn.Module):
    def __init__(self):
    # 直接给TVLoss_weight赋值为1了
        super(L_TV,self).__init__()
    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return 2*(h_tv/count_h+w_tv/count_w)/batch_size

# input=vi(RGB),色恒度损失
class Sa_Loss(nn.Module):
    def __init__(self):
        super(Sa_Loss, self).__init__()
    def forward(self, x ):
        b,c,h,w = x.shape
        r,g,b = torch.split(x, 1, dim=1)
        mean_rgb = torch.mean(x, [2,3], keepdim=True)
        mr,mg,mb = torch.split(mean_rgb, 1, dim=1)
        Dr = r-mr
        Dg = g-mg
        Db = b-mb
        k = torch.pow(torch.pow(Dr,2) + torch.pow(Db,2) + torch.pow(Dg,2),0.5)
        k = torch.mean(k)
        return k