import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.nn.functional as F


# attention fusion strategy, average based on weight maps
def attention_fusion_weight(tensor1, tensor2):
    f_channel = channel_fusion(tensor1, tensor2)
    f_spatial = spatial_fusion(tensor1, tensor2)
    tensor_f = (f_channel + f_spatial) / 2
    return tensor_f

def channel_fusion(tensor1, tensor2):
    # global max pooling
    shape = tensor1.size()
    global_p1m = F.max_pool2d(tensor1, kernel_size=shape[2:])
    global_p2m = 1.2*(F.max_pool2d(tensor2, kernel_size=shape[2:]))

    global_p_w1m=torch.exp(global_p1m) / (torch.exp(global_p2m) + torch.exp(global_p1m) + 0.000001)
    global_p_w2m=torch.exp(global_p2m) / (torch.exp(global_p2m) + torch.exp(global_p1m) + 0.000001)
    global_p_w1m = global_p_w1m.repeat(1, 1, shape[2], shape[3])
    global_p_w2m = global_p_w2m.repeat(1, 1, shape[2], shape[3])

    tensor_f = global_p_w1m*tensor1+global_p_w2m*tensor2
    return tensor_f


def spatial_fusion(tensor1, tensor2):
    # get weight map, soft-max
    shape = tensor1.size()
    spatial1m = []
    spatial2m = []
    spatial1m,_ = tensor1.max(dim=1, keepdim=True)
    spatial2m,_ = tensor2.max(dim=1, keepdim=True)
    
    spatial_w1m = torch.exp(spatial1m) / (torch.exp(spatial1m) + torch.exp(spatial2m) + 0.000001)
    spatial_w2m = torch.exp(spatial2m) / (torch.exp(spatial1m) + torch.exp(spatial2m) + 0.000001)
    spatial_w1m = spatial_w1m.repeat(1, shape[1], 1, 1)
    spatial_w2m = spatial_w2m.repeat(1, shape[1], 1, 1)

    tensor_f = spatial_w1m*tensor1+spatial_w2m*tensor2
    return tensor_f



