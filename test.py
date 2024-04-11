import os
from torch.autograd import Variable
import argparse
import time
import logging
import os.path as osp
import torch
import dataloader
import model
from logger import setup_logger
from torch.utils.data import DataLoader
from dataloader import rgb2ycbcr,ycbcr2rgb
from torchvision import transforms
from tqdm import tqdm
from thop import profile
import warnings
warnings.filterwarnings('ignore')


def run_enhance_gpu():
    total_time1=0
    enhance_model_path = osp.join(os.getcwd(),'checkpoint/enhancement_model.pth')
    enhanced_dir=osp.join(os.getcwd(),'test/LLVIP/vi_en/')
    os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)
    enhancemodel = model.luminance_adjustment()
    enhancemodel = enhancemodel.cuda()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(enhance_model_path))
    test_dataset = dataloader.enhance_dataset_loader_test(osp.join(os.getcwd(),'test/LLVIP/'))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    test_tqdm=tqdm(test_loader,total=len(test_loader))
    with torch.no_grad():
        for images_vis, name in test_tqdm:
            images_vis = Variable(images_vis)
            images_vis = images_vis.cuda()
            start_time = time.time()
            _,enhanced_image,_ = enhancemodel(images_vis)
            enhanced_image = enhanced_image.cpu().numpy() #转为array
            for k in range(len(name)):
                image = enhanced_image[k, :, :, :] #丢掉第0维
                image = image.squeeze()
                image=torch.tensor(image).to(images_vis.device)
                image=transforms.ToPILImage()(image)
                save_path = os.path.join(enhanced_dir,name[k])
                image.save(save_path)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time1 += elapsed_time

    average_time1 = total_time1 / 50
    print("Average time:", average_time1, "s")


def run_enhance_cpu():
    total_time1=0
    enhance_model_path = osp.join(os.getcwd(),'checkpoint/enhancement_model.pth')
    enhanced_dir=osp.join(os.getcwd(),'test/LLVIP/vi_en')
    os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)
    enhancemodel = model.luminance_adjustment()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(enhance_model_path, map_location=torch.device('cpu')))
    test_dataset = dataloader.enhance_dataset_loader_test(osp.join(os.getcwd(),'test/LLVIP/'))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    test_tqdm=tqdm(test_loader,total=len(test_loader))
    with torch.no_grad():
        for images_vis, name in test_tqdm:
            start_time = time.time()
            _,enhanced_image,_ = enhancemodel(images_vis)
            enhanced_image = enhanced_image.cpu().numpy() #转为array
            for k in range(len(name)):
                image = enhanced_image[k, :, :, :] #丢掉第0维
                image = image.squeeze()
                image=torch.tensor(image)
                image=transforms.ToPILImage()(image)
                save_path = os.path.join(enhanced_dir,name[k])
                image.save(save_path)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time1 += elapsed_time

    average_time1 = total_time1 / 50
    print("Average time:", average_time1, "s")

# print('==> Building model..')
# Amodel = model.luminance_adjustment()
# input = torch.randn(1, 3, 224, 224)
# flops, params = profile(Amodel, (input,))
# print('flops: %.3f G, params: %.3f M' % (flops / 1e9, params / 1e6))

def run_fusion_gpu():
    total_time=0
    fusion_model_path = osp.join(os.getcwd(),'checkpoint/fusion_model.pth')
    fusion_dir=osp.join(os.getcwd(),'test/LLVIP/')
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)
    fusionmodel = model.FusionNet()
    fusionmodel = fusionmodel.cuda()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path),False)
    enhance_model_path = osp.join(os.getcwd(),'checkpoint/enhancement_model.pth')
    enhanced_dir=osp.join(os.getcwd(),'test/LLVIP/vi_en/')
    os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)
    enhancemodel = model.luminance_adjustment()
    enhancemodel = enhancemodel.cuda()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(enhance_model_path))
    testdataset = dataloader.fusion_dataset_loader_test(osp.join(os.getcwd(),'test/LLVIP/'))
    testloader = DataLoader(
        dataset=testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    testtqdm=tqdm(testloader,total=len(testloader))

    with torch.no_grad():
        for images_vis,images_ir,name in testtqdm:
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            images_vis = images_vis.cuda()
            images_ir = images_ir.cuda()
            start_time = time.time()
            _,enhanced_image,_ = enhancemodel(images_vis)
            image_vis_en_ycbcr = rgb2ycbcr(enhanced_image)
            image_vis_en_y=dataloader.clahe(image_vis_en_ycbcr[:,0:1,:,:],image_vis_en_ycbcr.shape[0])
            _,_,_,Y_f = fusionmodel(image_vis_en_y,images_ir)
            fusion_ycbcr = torch.cat((Y_f,image_vis_en_ycbcr[:,1:2,:,:],image_vis_en_ycbcr[:,2:,:,:]),dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)

            ones = torch.ones_like(I_f)
            zeros = torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)
            I_f = I_f.cpu().numpy()
            for k in range(len(name)):
                image_vis_en_ycbcr = image_vis_en_ycbcr[k,:,:,:]
                fusion_ycbcr=fusion_ycbcr[k, :, :, :]
                image_I_f = I_f[k, :, :, :]
                image_I_f = image_I_f.squeeze()
                image_I_f=torch.tensor(image_I_f).to(images_vis.device)
                image_vis_en_ycbcr=torch.tensor(image_vis_en_ycbcr).to(images_vis.device)
                fusion_ycbcr=torch.tensor(fusion_ycbcr).to(images_vis.device)
                image_I_f=transforms.ToPILImage()(image_I_f)
                image_vis_en_ycbcr=transforms.ToPILImage()(image_vis_en_ycbcr)
                fusion_ycbcr=transforms.ToPILImage()(fusion_ycbcr)
                save_path1 = os.path.join(fusion_dir,'If', name[k])
                image_I_f.save(save_path1)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time

    average_time = total_time / 50
    print("Average time:", average_time, "s")

def run_fusion_cpu():
    total_time=0
    fusion_model_path = osp.join(os.getcwd(),'checkpoint/fusion_model.pth')
    fusion_dir=osp.join(os.getcwd(),'test/LLVIP/')
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)
    fusionmodel = model.FusionNet()
    # fusionmodel = fusionmodel.cuda()  # 移除.cuda()调用
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path, map_location=torch.device('cpu')))
    enhance_model_path = osp.join(os.getcwd(),'checkpoint/enhancement_model.pth')
    enhanced_dir=osp.join(os.getcwd(),'test/LLVIP/vi_en/')
    os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)
    enhancemodel = model.luminance_adjustment()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(enhance_model_path, map_location=torch.device('cpu')))
    testdataset = dataloader.fusion_dataset_loader_test(osp.join(os.getcwd(),'test/LLVIP/'))
    testloader = DataLoader(
        dataset=testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        drop_last=False,
    )
    testtqdm=tqdm(testloader,total=len(testloader))

    with torch.no_grad():
        for images_vis,images_ir,name in testtqdm:
            images_vis = Variable(images_vis)
            images_ir = Variable(images_ir)
            start_time = time.time()
            _,enhanced_image,_ = enhancemodel(images_vis)
            image_vis_en_ycbcr = rgb2ycbcr(enhanced_image)
            image_vis_en_y=dataloader.clahe(image_vis_en_ycbcr[:,0:1,:,:],image_vis_en_ycbcr.shape[0])
            _,_,_,Y_f = fusionmodel(image_vis_en_y,images_ir)
            fusion_ycbcr = torch.cat((Y_f,image_vis_en_ycbcr[:,1:2,:,:],image_vis_en_ycbcr[:,2:,:,:]),dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)

            ones = torch.ones_like(I_f)
            zeros = torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)
            I_f = I_f.numpy()  # 移至CPU上后，使用numpy()方法获取数据
            for k in range(len(name)):
                image_vis_en_ycbcr = image_vis_en_ycbcr[k,:,:,:].cpu()
                fusion_ycbcr=fusion_ycbcr[k, :, :, :].cpu()
                image_I_f = I_f[k, :, :, :]
                image_I_f = image_I_f.squeeze()
                image_I_f=torch.tensor(image_I_f)
                image_vis_en_ycbcr=torch.tensor(image_vis_en_ycbcr)
                fusion_ycbcr=torch.tensor(fusion_ycbcr)
                image_I_f=transforms.ToPILImage()(image_I_f)
                image_vis_en_ycbcr=transforms.ToPILImage()(image_vis_en_ycbcr)
                fusion_ycbcr=transforms.ToPILImage()(fusion_ycbcr)
                save_path1 = os.path.join(fusion_dir,'If', name[k])
                image_I_f.save(save_path1)
                end_time = time.time()
                elapsed_time = end_time - start_time
                total_time += elapsed_time

    average_time = total_time / 50
    print("Average time:", average_time, "s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test for enhancement and fusion')
    parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    if args.use_gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        run_enhance_gpu()
        print("|Enhancement Image Sucessfully~!")
        run_fusion_gpu()
        print("|啊哈哈|Fusion Image Sucessfully~!")
    else:
        run_enhance_cpu()  
        print("|Enhancement Image Sucessfully~!")
        run_fusion_cpu()
        print("|啊哈哈|Fusion Image Sucessfully~!")
    print("————————————————————————————————————————————")
    print("Test Done!")