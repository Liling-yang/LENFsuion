import numpy as np
from scipy.signal import convolve2d
import math

def sobel_fn(x):
    # Sobel operators
    vtemp = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]) / 8
    htemp = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]) / 8

    a, b = htemp.shape
    x_ext = per_extn_im_fn(x, a)
    p, q = x_ext.shape
    gv = np.zeros((p - 2, q - 2))
    gh = np.zeros((p - 2, q - 2))
    gv = convolve2d(x_ext, vtemp, mode='valid')
    gh = convolve2d(x_ext, htemp, mode='valid')
    # for ii in range(1, p - 1):
    #     for jj in range(1, q - 1):
    #         gv[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * vtemp)
    #         gh[ii - 1, jj - 1] = np.sum(x_ext[ii - 1:ii + 2, jj - 1:jj + 2] * htemp)

    return gv, gh


def per_extn_im_fn(x, wsize):
    """
    Periodic extension of the given image in 4 directions.

    xout_ext = per_extn_im_fn(x, wsize)

    Periodic extension by (wsize-1)/2 on all 4 sides.
    wsize should be odd.

    Example:
        Y = per_extn_im_fn(X, 5);    % Periodically extends 2 rows and 2 columns in all sides.
    """

    hwsize = (wsize - 1) // 2  # Half window size excluding centre pixel.

    p, q = x.shape
    xout_ext = np.zeros((p + wsize - 1, q + wsize - 1))
    xout_ext[hwsize: p + hwsize, hwsize: q + hwsize] = x

    # Row-wise periodic extension.
    if wsize - 1 == hwsize + 1:
        xout_ext[0: hwsize, :] = xout_ext[2, :].reshape(1, -1)
        xout_ext[p + hwsize: p + wsize - 1, :] = xout_ext[-3, :].reshape(1, -1)

    # Column-wise periodic extension.
    xout_ext[:, 0: hwsize] = xout_ext[:, 2].reshape(-1, 1)
    xout_ext[:, q + hwsize: q + wsize - 1] = xout_ext[:, -3].reshape(-1, 1)

    return xout_ext

def get_Nabf(I1, I2, f):
    # Parameters for Petrovic Metrics Computation.
    Td=2
    wt_min=0.001
    P=1
    Lg=1.5
    Nrg=0.9999
    kg=19
    sigmag=0.5
    Nra=0.9995
    ka=22
    sigmaa=0.5

    xrcw = f.astype(np.float64)
    x1 = I1.astype(np.float64)
    x2 = I2.astype(np.float64)

    # Edge Strength & Orientation.
    gvA,ghA=sobel_fn(x1)
    gA=np.sqrt(ghA**2+gvA**2)

    gvB,ghB=sobel_fn(x2)
    gB=np.sqrt(ghB**2+gvB**2)

    gvF,ghF=sobel_fn(xrcw)
    gF=np.sqrt(ghF**2+gvF**2)

    # Relative Edge Strength & Orientation.
    gAF=np.zeros(gA.shape)
    gBF=np.zeros(gB.shape)
    aA=np.zeros(ghA.shape)
    aB=np.zeros(ghB.shape)
    aF=np.zeros(ghF.shape)
    p,q=xrcw.shape
    maskAF1 = (gA == 0) | (gF == 0)
    maskAF2 = (gA > gF)
    gAF[~maskAF1] = np.where(maskAF2, gF / gA, gA / gF)[~maskAF1]
    maskBF1 = (gB == 0) | (gF == 0)
    maskBF2 = (gB > gF)
    gBF[~maskBF1] = np.where(maskBF2, gF / gB, gB / gF)[~maskBF1]
    aA = np.where((gvA == 0) & (ghA == 0), 0, np.arctan(gvA / ghA))
    aB = np.where((gvB == 0) & (ghB == 0), 0, np.arctan(gvB / ghB))
    aF = np.where((gvF == 0) & (ghF == 0), 0, np.arctan(gvF / ghF))

    aAF=np.abs(np.abs(aA-aF)-np.pi/2)*2/np.pi
    aBF=np.abs(np.abs(aB-aF)-np.pi/2)*2/np.pi

    QgAF = Nrg / (1 + np.exp(-kg * (gAF - sigmag)))
    QaAF = Nra / (1 + np.exp(-ka * (aAF - sigmaa)))
    QAF = np.sqrt(QgAF * QaAF)
    QgBF = Nrg / (1 + np.exp(-kg * (gBF - sigmag)))
    QaBF = Nra / (1 + np.exp(-ka * (aBF - sigmaa)))
    QBF = np.sqrt(QgBF * QaBF)

    wtA = wt_min * np.ones((p, q))
    wtB = wt_min * np.ones((p, q))
    cA = np.ones((p, q))
    cB = np.ones((p, q))
    wtA = np.where(gA >= Td, cA * gA ** Lg, 0)
    wtB = np.where(gB >= Td, cB * gB ** Lg, 0)

    wt_sum = np.sum(wtA + wtB)
    QAF_wtsum = np.sum(QAF * wtA) / wt_sum  # Information Contributions of A.
    QBF_wtsum = np.sum(QBF * wtB) / wt_sum  # Information Contributions of B.
    QABF = QAF_wtsum + QBF_wtsum  # QABF=sum(sum(QAF.*wtA+QBF.*wtB))/wt_sum -> Total Fusion Performance.


    Qdelta = np.abs(QAF - QBF)
    QCinfo = (QAF + QBF - Qdelta) / 2
    QdeltaAF = QAF - QCinfo
    QdeltaBF = QBF - QCinfo
    QdeltaAF_wtsum = np.sum(QdeltaAF * wtA) / wt_sum
    QdeltaBF_wtsum = np.sum(QdeltaBF * wtB) / wt_sum
    QdeltaABF = QdeltaAF_wtsum + QdeltaBF_wtsum  # Total Fusion Gain.
    QCinfo_wtsum = np.sum(QCinfo * (wtA + wtB)) / wt_sum
    QABF11 = QdeltaABF + QCinfo_wtsum  # Total Fusion Performance.

    rr = np.zeros((p, q))
    rr = np.where(gF <= np.minimum(gA, gB), 1, 0)


    LABF = np.sum(rr * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum

    na1 = np.where((gF > gA) & (gF > gB), 2 - QAF - QBF, 0)
    NABF1 = np.sum(na1 * (wtA + wtB)) / wt_sum

    # Fusion Artifacts (NABF) changed by B. K. Shreyamsha Kumar.

    na = np.where((gF > gA) & (gF > gB), 1, 0)
    NABF = np.sum(na * ((1 - QAF) * wtA + (1 - QBF) * wtB)) / wt_sum
    return NABF