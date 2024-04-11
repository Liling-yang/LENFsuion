import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import glob
import random
import cv2
import os.path as osp
from torchvision import transforms
from PIL import Image

to_tensor = transforms.Compose([transforms.ToTensor()])
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
random.seed(1143)

def populate_train_list(lowlight_images_path):
    image_list_lowlight = glob.glob(lowlight_images_path + "*.png")
    train_list = image_list_lowlight
    random.shuffle(train_list)
    return train_list

class lowlight_loader(data.Dataset):
    def __init__(self, lowlight_images_path):
        self.train_list = populate_train_list(lowlight_images_path) 
        self.size = 256
        self.data_list = self.train_list
        print("Total training examples:", len(self.train_list))

    def __getitem__(self, index):
        data_lowlight_path = self.data_list[index]
        data_lowlight = Image.open(data_lowlight_path)
        data_lowlight = data_lowlight.resize((self.size,self.size), Image.ANTIALIAS) #等比例压缩图片
        data_lowlight = (np.asarray(data_lowlight)/255.0) 
        data_lowlight = torch.from_numpy(data_lowlight).float()
        return data_lowlight.permute(2,0,1)

    def __len__(self):
        return len(self.data_list)

def prepare_data_path(dataset_path):
    filenames = os.listdir(dataset_path)
    data_dir = dataset_path
    data = glob.glob(os.path.join(data_dir, "*.bmp"))
    data.extend(glob.glob(os.path.join(data_dir, "*.tif")))
    data.extend(glob.glob((os.path.join(data_dir, "*.jpg"))))
    data.extend(glob.glob((os.path.join(data_dir, "*.png"))))
    data.sort()
    filenames.sort()
    return data, filenames

class fusion_dataset_loader(data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(fusion_dataset_loader, self).__init__()
        self.size = 225 # patch_size
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        if split == 'train':
            data_dir_vis = os.path.join(os.getcwd(), 'train/vi')
            data_dir_ir = os.path.join(os.getcwd(), 'train/ir')
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))

    def __getitem__(self, index):
        if self.split=='train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis=Image.open(vis_path)
            image_vis=image_vis.resize((self.size,self.size), Image.ANTIALIAS) #等比例压缩图片
            image_vis = np.array(image_vis)
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/255.0)

            image_inf=Image.open(ir_path)
            image_inf=image_inf.resize((self.size,self.size), Image.ANTIALIAS) #等比例压缩图片
            image_inf = np.array(image_inf)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            image_vis = torch.tensor(image_vis)
            image_ir = torch.tensor(image_ir)
            return (image_vis,image_ir,)
        
    def __len__(self):
        return self.length

class enhance_dataset_loader(data.Dataset):
    def __init__(self, split, ir_path=None, vi_path=None):
        super(enhance_dataset_loader, self).__init__()
        self.size=225 # patch_size
        assert split in ['train', 'val', 'test'], 'split must be "train"|"val"|"test"'
        if split == 'train':
            data_dir_vis = os.path.join(os.getcwd(), 'train/vi')#增强前的vi
            data_dir_ir = os.path.join(os.getcwd(), 'train/ir')
            self.filepath_vis, self.filenames_vis = prepare_data_path(data_dir_vis)
            self.filepath_ir, self.filenames_ir = prepare_data_path(data_dir_ir)
            self.split = split
            self.length = min(len(self.filenames_vis), len(self.filenames_ir))
    def __getitem__(self, index):
        if self.split == 'train':
            vis_path = self.filepath_vis[index]
            ir_path = self.filepath_ir[index]
            image_vis=Image.open(vis_path)
            image_vis=image_vis.resize((self.size,self.size), Image.ANTIALIAS) #等比例压缩图片
            image_vis = np.array(image_vis)
            image_vis = (np.asarray(Image.fromarray(image_vis), dtype=np.float32).transpose((2, 0, 1))/255.0)
            image_inf=Image.open(ir_path)
            image_inf=image_inf.resize((self.size,self.size), Image.ANTIALIAS) #等比例压缩图片
            image_inf = np.array(image_inf)
            image_ir = np.asarray(Image.fromarray(image_inf), dtype=np.float32) / 255.0
            image_ir = np.expand_dims(image_ir, axis=0)
            return (torch.tensor(image_vis),torch.tensor(image_ir),)
    def __len__(self):
        return self.length

# for LAN test and eval
class enhance_dataset_loader_test(data.Dataset):
    def __init__(self, data_dir, transform=to_tensor):
        super().__init__()
        temp_path = os.path.join(data_dir,'vi')
        self.vis_path=temp_path
        self.name_list = os.listdir(self.vis_path)  # 获得子目录下的图片的名称
        self.transform = transform
    def __getitem__(self, index):
        name = self.name_list[index]  # 获得当前图片的名称
        image_vis = Image.open(os.path.join(self.vis_path, name))
        image_vis = self.transform(image_vis)
        return image_vis, name
    def __len__(self):
        return len(self.name_list)
    
class fusion_dataset_loader_eval(data.Dataset):
    def __init__(self,i,data_dir,transform=to_tensor):
        super().__init__()
        dirname=os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'vi_en':
                self.vis_path=osp.join(temp_path,str(i+1))
        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform
    def __getitem__(self,index):
        name = self.name_list[index]  # 获得当前图片的名称
        inf_image = Image.open(os.path.join(self.inf_path, name))
        vis_image = Image.open(os.path.join(self.vis_path, name))
        ir_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        return vis_image, ir_image, name
    def __len__(self):
        return len(self.name_list)

class fusion_dataset_loader_test(data.Dataset):
    def __init__(self,data_dir,transform=to_tensor):
        super().__init__()
        dirname=os.listdir(data_dir)
        for sub_dir in dirname:
            temp_path = os.path.join(data_dir, sub_dir)
            if sub_dir == 'ir':
                self.inf_path = temp_path  # 获得红外路径
            elif sub_dir == 'vi':
                self.vis_path=osp.join(temp_path)
        self.name_list = os.listdir(self.inf_path)  # 获得子目录下的图片的名称
        self.transform = transform
    def __getitem__(self,index):
        name = self.name_list[index]  # 获得当前图片的名称
        inf_image = Image.open(os.path.join(self.inf_path, name))
        vis_image = Image.open(os.path.join(self.vis_path, name))
        ir_image = self.transform(inf_image)
        vis_image = self.transform(vis_image)
        return vis_image, ir_image, name
    def __len__(self):
        return len(self.name_list)

def rgb2ycbcr(input_im):
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    R = im_flat[:, 0]
    G = im_flat[:, 1]
    B = im_flat[:, 2]
    Y = 0.299 * R + 0.587 * G + 0.114 * B
    Cr = (R - Y) * 0.713 + 0.5
    Cb = (B - Y) * 0.564 + 0.5

    Y=torch.clamp(Y, min=0., max=1.0)
    Cr=torch.clamp(Cr, min=0., max=1.0)
    Cb=torch.clamp(Cb, min=0., max=1.0)
    Y = torch.unsqueeze(Y, 1)#升维
    Cr = torch.unsqueeze(Cr, 1)
    Cb = torch.unsqueeze(Cb, 1)
    # temp = torch.cat((Y, Cr, Cb), dim=1) CPU版本
    temp = torch.cat((Y, Cr, Cb), dim=1).cuda()
    out = (   temp.reshape(
            list(input_im.size())[0],
            list(input_im.size())[2],
            list(input_im.size())[3],
            3,)
        .transpose(1, 3)
        .transpose(2, 3))
    return out

def ycbcr2rgb(input_im):
    B, C, W, H = input_im.shape
    im_flat = input_im.transpose(1, 3).transpose(1, 2).reshape(-1, 3)
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]])
    bias = torch.tensor([0.0 / 255, -0.5, -0.5])
    mat = torch.tensor([[1.0, 1.0, 1.0], [1.403, -0.714, 0.0], [0.0, -0.344, 1.773]]).cuda()
    bias = torch.tensor([0.0 / 255, -0.5, -0.5]).cuda()
    temp = (im_flat + bias).mm(mat).cuda()
    out = temp.reshape(B, W, H, C).transpose(1, 3).transpose(2, 3).cuda()
    out = torch.clamp(out, min=0., max=1.0)
    return out

def clahe(image, batch_size):
    image = image.cpu().detach().numpy()  # 移至CPU中（转为Tensor），再转为NumPy
    results = []
    for i in range(batch_size):
        img = np.squeeze(image[i:i+1, :, :, :])
        out = np.array(cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX), dtype='uint8')
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(4, 4))
        result = clahe.apply(out)[np.newaxis][np.newaxis]
        results.append(result)
    results = np.concatenate(results, axis=0)
    image_hist = (results / 255.0).astype(np.float32)
    image_hist = torch.from_numpy(image_hist).cuda()
    return image_hist