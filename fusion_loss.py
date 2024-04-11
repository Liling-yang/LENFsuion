import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn as nn
import torch.nn.functional as F
import model
import enhancement_loss
from dataloader import rgb2ycbcr,ycbcr2rgb
from math import exp



class fusionloss(nn.Module):
    def __init__(self):
        super(fusionloss, self).__init__()
        self.sobelconv=Sobelxy()
        self.mse_loss = nn.MSELoss()
        self.fuionnet = model.FusionNet().cuda()
        self.angle=angle()
        self.L_color=enhancement_loss.L_color()
        
    # vi_en是三通道图片，ir是单通道灰度图，y_f是I_f的Y通道图，I_f由YCbCr组成
    def forward(self,vi_en_y,vi_en,ir,y_f,I_f):#vi_en是RGB的
        vi_en_y_gard=self.sobelconv(vi_en_y)
        ir_gard=self.sobelconv(ir)
        y_f_grad=self.sobelconv(y_f)
        max_grad=torch.max(vi_en_y_gard,ir_gard)
        grad_loss = F.l1_loss(max_grad,y_f_grad)
     
        # MSE intensity Loss MSE强度损失函数
        max_init = torch.max(ir,vi_en_y)
        image_loss = F.l1_loss(y_f, max_init)
        color_loss=torch.mean(self.L_color(I_f))
        total_loss = 120*image_loss + 10*grad_loss + 0.05*color_loss
        return total_loss,image_loss,grad_loss,color_loss

#用两个向量间的余弦值求的arccos
class angle(nn.Module):
    def __init__(self):
        super(angle,self).__init__()
    def forward(self,a,b):
        vector = torch.mul(a,b) # 点乘
        up = torch.sum(vector)
        down = torch.sqrt(torch.sum(torch.pow(a,2))) * torch.sqrt(torch.sum(torch.pow(b,2)))
        theta = torch.acos(up/down)  # 弧度制
        return theta
    
class Sobelxy(nn.Module):
    def __init__(self):
        super(Sobelxy, self).__init__()
        kernelx = [[-1, 0, 1],
                  [-2,0 , 2],
                  [-1, 0, 1]]
        kernely = [[1, 2, 1],
                  [0,0 , 0],
                  [-1, -2, -1]]
        kernelx = torch.FloatTensor(kernelx).unsqueeze(0).unsqueeze(0)
        kernely = torch.FloatTensor(kernely).unsqueeze(0).unsqueeze(0)
        self.weightx = nn.Parameter(data=kernelx, requires_grad=False).cuda()
        self.weighty = nn.Parameter(data=kernely, requires_grad=False).cuda()
    def forward(self,x):
        sobelx=F.conv2d(x, self.weightx, padding=1)
        sobely=F.conv2d(x, self.weighty, padding=1)
        return torch.abs(sobelx)+torch.abs(sobely)

def get_per(img):
	fro_2_norm = torch.sum(torch.pow(img,2),dim=[1,2,3])
	loss=fro_2_norm / (225.0*225.0)
	return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel=1):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)                            
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)    
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()  
    return window

def mssim(img1, img2, window_size=11):
    max_val = 255
    min_val = 0
    L = max_val - min_val
    padd = window_size // 2
    (_, channel, height, width) = img1.size()

    window = create_window(window_size, channel=channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=padd, groups=channel)
    mu2 = F.conv2d(img2, window, padding=padd, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=padd, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=padd, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=padd, groups=channel) - mu1_mu2

    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    v1 = 2.0 * sigma12 + C2
    v2 = sigma1_sq + sigma2_sq + C2
    cs = torch.mean(v1 / v2) 
    ssim_map = ((2 * mu1_mu2 + C1) * v1) / ((mu1_sq + mu2_sq + C1) * v2)
    ret = ssim_map
    return ret

def std(img,  window_size=9):
    padd = window_size // 2
    (_, channel, height, width) = img.size()
    window = create_window(window_size, channel=channel).to(img.device)
    mu = F.conv2d(img, window, padding=padd, groups=channel)
    mu_sq = mu.pow(2)
    sigma1 = F.conv2d(img * img, window, padding=padd, groups=channel) - mu_sq
    return sigma1

def final_ssim(img_ir, img_vis, img_fuse):

    ssim_ir = mssim(img_ir, img_fuse)
    ssim_vi = mssim(img_vis, img_fuse)

    std_ir = std(img_ir)
    std_vi = std(img_vis)

    zero = torch.zeros_like(std_ir)
    one = torch.ones_like(std_vi)

    map1 = torch.where((std_ir - std_vi) > 0, one, zero)
    map2 = torch.where((std_ir - std_vi) >= 0, zero, one)

    ssim = map1 * ssim_ir + map2 * ssim_vi
    return ssim.mean()