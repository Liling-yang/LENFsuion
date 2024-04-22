# LENFusion

LENFusion: A Joint Low-Light Enhancement and Fusion Network for Nighttime Infrared and Visible Image Fusion

This is official code of "[LENFusion: A Joint Low-Light Enhancement and Fusion Network for Nighttime Infrared and Visible Image Fusion](https://ieeexplore.ieee.org/abstract/document/10504357)".

## Before Train

> conda env create -f LENFusion.yaml

## Train

During training, **eval** is used for evaluation, and **model** is used to store the training weight.

The code has two training datasets in the **train** folder:
1. **LFN_traingdata** is the training dataset of Luminance Feedback Network (LFN). Links: [google drive](https://drive.google.com/file/d/16VLXA-aOtD_TJaVFP9qEW-2Fa-05PJW2/view?usp=drive_link) and [baidu](https://pan.baidu.com/s/1Fw6nPvlTv9A3vAOGd3D9Aw?pwd=vudc).
2.  **vi** and **ir** are the training dataset of the main networks (Luminance Adjustment Network and Re-enhancement Fusion Network, LAN and RFN). Links: [google drive](https://drive.google.com/file/d/19zx4yWi_T7skTIfaJKLAmsbPjKNSzSgX/view?usp=drive_link) and [baidu](https://pan.baidu.com/s/1Q81kiIrCVACC703i1r_osQ?pwd=x6gf ).

## Test

Download **checkpoint** at the links: [google drive](https://drive.google.com/drive/folders/1RJEd-PLDZUq8NnE3T-KhACsALlR2uRyS?usp=drive_link) and [baidu](https://pan.baidu.com/s/13ncLAdDAjIXIyZk5drq_ZQ?pwd=g7nn).
The test data from the [KAIST](https://github.com/SoonminHwang/rgbt-ped-detection), [LLVIP](https://github.com/bupt-ai-cz/LLVIP), [MSRS](https://github.com/Linfeng-Tang/MSRS), [TNO](https://figshare.com/articles/dataset/TNO_Image_Fusion_Dataset/1008029) datasets.



## The Environment

>numpy=1.26.0
>
>opencv-python=4.8.1.78
>
>python=3.10.13
>
>torch=1.12.1
>
>torchvision=0.13.1


## If this work is helpful to you, please cite it as：
>@article{Chen2024LENFusion,
  title={LENFusion: A Joint Low-Light Enhancement and Fusion Network for Nighttime Infrared and Visible Image Fusion},
  author={Tang, Linfeng and Xiang, Xinyu and Zhang, Hao and Gong, Meiqi and Ma, Jiayi},
  journal={Chen, Jun and Yang, Liling and Liu, Wei and Tian, Xin and Ma, Jiayi},
  volume = {},
  pages = {},
  year = {2024}
}
