import os
import numpy as np
from scipy.signal import convolve2d
from Qabf import get_Qabf
from Nabf import get_Nabf
import math
from ssim import ssim, ms_ssim


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def EN_function(image_array):
    histogram, _ = np.histogram(image_array, bins=256, range=(0, 255))
    histogram = histogram / np.sum(histogram)
    entropy = -np.sum(histogram * np.log2(histogram + 1e-7))
    return entropy


def SF_function(image):
    RF = np.diff(image, axis=0)
    CF = np.diff(image, axis=1)
    SF = np.sqrt(np.mean(RF ** 2) + np.mean(CF ** 2))
    return SF


def SD_function(image_array):
    return np.std(image_array)


def PSNR_function(ir_img, vi_img, f_img):
    ir_img = ir_img.astype(np.float64)
    vi_img = vi_img.astype(np.float64)
    f_img = f_img.astype(np.float64)
    mse_ir = np.mean(((f_img - ir_img) / 255.0) ** 2)
    mse_vi = np.mean(((f_img - vi_img) / 255.0) ** 2)
    mse = (mse_ir + mse_vi) / 2
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr


# def MSE_function(A, B, F):
    # A, B, F = A / 255.0, B / 255.0, F / 255.0
    # MSE_AF = np.mean((F - A) ** 2)
    # MSE_BF = np.mean((F - B) ** 2)
    # MSE = 0.5 * (MSE_AF + MSE_BF)
    # return max(MSE, 0.0)

def MSE_function(ir_img, vi_img, f_img):
    ir_img = ir_img.astype(np.float64)
    vi_img = vi_img.astype(np.float64)
    f_img = f_img.astype(np.float64)
    mse_ir = np.mean(((f_img - ir_img) / 255.0) ** 2)
    mse_vi = np.mean(((f_img - vi_img) / 255.0) ** 2)
    mse = (mse_ir + mse_vi) / 2
    return mse


# def MI_function(A, B, F, gray_level=256):
#     def joint_histogram(im1, im2):
#         return np.histogram2d(im1.ravel(), im2.ravel(), bins=gray_level)[0] / (im1.size)

#     h_AF = joint_histogram(A, F)
#     h_BF = joint_histogram(B, F)
#     MI_AF = np.sum(h_AF * np.log2(h_AF + 1e-7))
#     MI_BF = np.sum(h_BF * np.log2(h_BF + 1e-7))
#     return MI_AF + MI_BF

def MI_function(A, B, F, gray_level=256):
    def joint_histogram(im1, im2):
        return np.histogram2d(im1.ravel(), im2.ravel(), bins=gray_level)[0] / (im1.size)
    
    def marginal_histogram(im):
        return np.histogram(im.ravel(), bins=gray_level)[0] / (im.size)
    
    h_AF = joint_histogram(A, F)
    h_BF = joint_histogram(B, F)
    
    P_A = marginal_histogram(A)
    P_B = marginal_histogram(B)
    P_F = marginal_histogram(F)
    
    # 计算 MI_AF
    MI_AF = np.sum(h_AF * np.log2((h_AF + 1e-7) / (P_A[:, None] * P_F[None, :] + 1e-7)))
    
    # 计算 MI_BF
    MI_BF = np.sum(h_BF * np.log2((h_BF + 1e-7) / (P_B[:, None] * P_F[None, :] + 1e-7)))
    
    return MI_AF + MI_BF


def fspecial_gaussian(shape, sigma):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    h /= np.sum(h)
    return h


def vifp_mscale(ref, dist):
    sigma_nsq = 2
    num = 0
    den = 0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        win = fspecial_gaussian((N, N), N / 5)

        if scale > 1:
            ref = convolve2d(ref, win, mode='valid')[::2, ::2]
            dist = convolve2d(dist, win, mode='valid')[::2, ::2]

        mu1 = convolve2d(ref, win, mode='valid')
        mu2 = convolve2d(dist, win, mode='valid')
        sigma1_sq = convolve2d(ref * ref, win, mode='valid') - mu1 ** 2
        sigma2_sq = convolve2d(dist * dist, win, mode='valid') - mu2 ** 2
        sigma12 = convolve2d(ref * dist, win, mode='valid') - mu1 * mu2

        g = sigma12 / (sigma1_sq + 1e-10)
        sv_sq = sigma2_sq - g * sigma12

        g[sigma1_sq < 1e-10] = 0
        sv_sq[sigma1_sq < 1e-10] = sigma2_sq[sigma1_sq < 1e-10]
        sv_sq[sv_sq <= 1e-10] = 1e-10

        num += np.sum(np.log10(1 + g ** 2 * sigma1_sq / (sv_sq + sigma_nsq)))
        den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))
    return num / den


def VIF_function(A, B, F):
    return vifp_mscale(A, F) + vifp_mscale(B, F)


def CC_function(A, B, F):
    rAF = np.corrcoef(A.ravel(), F.ravel())[0, 1]
    rBF = np.corrcoef(B.ravel(), F.ravel())[0, 1]
    return np.mean([rAF, rBF])


# def corr2(a, b):
#     return np.corrcoef(a.ravel(), b.ravel())[0, 1]
# def SCD_function(A, B, F):
#     return corr2(F - B, A) + corr2(F - A, B)

def SCD_function(ir_img, vi_img, f_img):
    ir_img = ir_img.astype(np.float64)
    vi_img = vi_img.astype(np.float64)
    f_img = f_img.astype(np.float64)
    ir_vi_diff = np.abs(ir_img - vi_img)
    ir_vi_sum = np.sum(ir_img) + np.sum(vi_img)
    ir_vi_scd = np.sum(ir_vi_diff) / (ir_vi_sum + 1e-6)

    f_ir_diff = np.abs(f_img - ir_img)
    f_vi_diff = np.abs(f_img - vi_img)
    f_ir_sum = np.sum(f_img) + np.sum(ir_img)
    f_vi_sum = np.sum(f_img) + np.sum(vi_img)
    
    f_ir_scd = np.sum(f_ir_diff) / (f_ir_sum + 1e-6)
    f_vi_scd = np.sum(f_vi_diff) / (f_vi_sum + 1e-6)
    scd = (f_ir_scd + f_vi_scd) / (ir_vi_scd + 1e-6)
    
    return scd


def Qabf_function(A, B, F):
    return get_Qabf(A, B, F)


def Nabf_function(A, B, F):
    return get_Nabf(A, B, F)


def AG_function(image):
    gradx, grady = np.gradient(image)
    s = np.sqrt((gradx ** 2 + grady ** 2) / 2)
    return np.mean(s)


def SSIM_function(A, B, F):
    ssim_A = ssim(A, F)
    ssim_B = ssim(B, F)
    return 0.5 * (ssim_A + ssim_B)


def MS_SSIM_function(A, B, F):
    ms_ssim_A = ms_ssim(A, F)
    ms_ssim_B = ms_ssim(B, F)
    return 0.5 * (ms_ssim_A + ms_ssim_B)
