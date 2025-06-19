import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import numpy as np
from scipy.fftpack import dctn
from scipy.signal import convolve2d
from scipy.ndimage import sobel, generic_gradient_magnitude


def analysis_fmi(ima, imb, imf, feature, w):
    ima = np.double(ima)
    imb = np.double(imb)
    imf = np.double(imf)

    # Feature Extraction
    if feature == 'none':  # Raw pixels (no feature extraction)
        aFeature = ima
        bFeature = imb
        fFeature = imf
    elif feature == 'gradient':  # Gradient
        aFeature = generic_gradient_magnitude(ima, sobel)
        bFeature = generic_gradient_magnitude(imb, sobel)
        fFeature = generic_gradient_magnitude(imf, sobel)
    elif feature == 'edge':  # Edge
        aFeature = np.double(sobel(ima) > w)
        bFeature = np.double(sobel(imb) > w)
        fFeature = np.double(sobel(imf) > w)
    elif feature == 'dct':  # DCT
        aFeature = dctn(ima, type=2, norm='ortho')
        bFeature = dctn(imb, type=2, norm='ortho')
        fFeature = dctn(imf, type=2, norm='ortho')
    elif feature == 'wavelet':  # Discrete Meyer wavelet
        raise NotImplementedError('Wavelet feature extraction not yet implemented in Python!')
    else:
        raise ValueError(
            "Please specify a feature extraction method among 'gradient', 'edge', 'dct', 'wavelet', or 'none' (raw pixels)!")

    m, n = aFeature.shape
    w = w // 2
    fmi_map = np.ones((m - 2 * w, n - 2 * w))
