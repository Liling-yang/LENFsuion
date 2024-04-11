import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"#指定GPU运行
from torch.autograd import Variable
import argparse
import datetime
import time
import math
import logging
import os.path as osp
import torch
import dataloader
import model
import enhancement_loss
from fusion_loss import fusionloss,final_ssim
from logger import setup_logger
from torch.utils.data import DataLoader
from LFN_model import luminance_feedback
from dataloader import rgb2ycbcr,ycbcr2rgb
from torchvision import transforms
from tqdm import tqdm
from thop import profile
import warnings
warnings.filterwarnings('ignore')

def weights_init(m):
    # 获得nn.module的名字，初始化权重
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        # mean=0.0,std=0.02
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('conv') != -1:
        m.weight.data.normal_(0.0, 0.02)

def train_fusion(i, logger=None):#再增强与融合网络的训练函数
    modelpth = './model'
    modelpth = osp.join(modelpth,str(i+1))
    os.makedirs(modelpth, mode=0o777, exist_ok=True)
    fusion_batch_size = 5
    n_workers = 4
    
    ds = dataloader.fusion_dataset_loader('train') # load training data
    dl = torch.utils.data.DataLoader(ds, batch_size=fusion_batch_size,shuffle=True,num_workers=n_workers,pin_memory=False)
    # dl是增强之后的vi_en和ir

    net = model.FusionNet()
    if i==0:
        net.apply(weights_init)
    if i>0: #第二次训练加载上一次训练的字典
        load_path = './model'
        load_path = osp.join(load_path,str(i),'fusion_model.pth')
        net.load_state_dict(torch.load(load_path))
        print('Load Pre-trained Fusion Model:{}!'.format(load_path)) #加载之前训练的fusion模型
    net.cuda()
    net.eval()
    net.train()

    enhancemodel = model.luminance_adjustment().cuda()#亮度调整网络
    lr_start=1e-4
    optim = torch.optim.Adam(net.parameters(), lr=lr_start, weight_decay=0.0001)
    criteria_fusion=fusionloss()
    st = glob_st = time.time()
    epoch=20
    grad_step=5.0 # gradient accumulation to increase batch_size
    dl.n_iter=len(dl)

    for epo in range(0,epoch):
        lr_decay=0.75
        lr_this_epo=lr_start*lr_decay**((epo/5)+1)#迭代变更初始学习率
        for param_group in optim.param_groups:
            param_group['lr'] = lr_this_epo
        for it,(image_vis,image_ir) in enumerate(dl):
            net.train()
            image_vis=Variable(image_vis,requires_grad=True).cuda()
            _,image_vis_en,_=enhancemodel(image_vis)
            image_vis_en_ycbcr = rgb2ycbcr(image_vis_en)
            image_ir = Variable(image_ir,requires_grad=True).cuda()
            vi_en_y=dataloader.clahe(image_vis_en_ycbcr[:,0:1,:,:],image_vis_en_ycbcr.shape[0])
            _,_,_,Y_f = net(vi_en_y,image_ir)
            fusion_ycbcr = torch.cat((Y_f,image_vis_en_ycbcr[:,1:2,:,:],image_vis_en_ycbcr[:,2:,:,:]),dim=1)
            I_f = ycbcr2rgb(fusion_ycbcr)
            
            ones = torch.ones_like(I_f)
            zeros = torch.zeros_like(I_f)
            I_f = torch.where(I_f > ones, ones, I_f)
            I_f = torch.where(I_f < zeros, zeros, I_f)

            loss_fusion,loss_image,loss_grad,loss_color = criteria_fusion(vi_en_y,image_vis_en,image_ir,Y_f,I_f)

            ssim_loss=0
            ssim_loss_temp=1-final_ssim(image_ir,vi_en_y,Y_f)
            ssim_loss += ssim_loss_temp
            ssim_loss /= len(Y_f)
            loss_fusion=loss_fusion+30*ssim_loss
            loss_fusion.backward()
            for name, param in net.named_parameters():
                if param.grad is None:
                    print(name, param.grad_fn)
            if grad_step>1:
                loss_fusion=loss_fusion/grad_step
            if (it+1)%grad_step==0:
                optim.step()
                optim.zero_grad()
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            now_it = dl.n_iter * epo + it + 1
            if (it + 1) % 50 == 0: 
                lr = optim.param_groups[0]['lr']
                eta = int((dl.n_iter * epoch - now_it)* (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                msg = ', '.join(
                    ['step: {it}/{max_it}',
                            'loss_fusion:{loss_fusion:.4f}\n',
                            'loss_image: {loss_image:.4f}',
                            'loss_grad: {loss_grad:4f}',
                            'loss_color: {loss_color:4f}',
                            'loss_ssim:{loss_ssim:4f}',
                            'eta: {eta}',
                            'time: {time:.4f}',]).format(
                        it=now_it,max_it=dl.n_iter * epoch,lr=lr,
                        loss_fusion=loss_fusion,loss_image=loss_image,
                        loss_grad=loss_grad,loss_color=loss_color,
                        loss_ssim=ssim_loss,eta=eta,time=t_intv,)
                logger.info(msg)
                st = ed
    save_pth = osp.join(modelpth, 'fusion_model.pth')
    state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
    torch.save(state, save_pth)
    logger.info('Fusion Model Training done~, The Model is saved to: {}'.format(save_pth))
    logger.info('\n')

def train_enhancement(num, logger=None):
    enhance_batch_size = 4
    n_workers = 4
    lr_start=2*1e-5
    modelpth = './model'
    modelpth = osp.join(modelpth,str(num+1))
    enhancemodel = model.luminance_adjustment().cuda()
    enhancemodel.apply(weights_init)
    enhancemodel.eval()
    enhancemodel.train()
    optimizer = torch.optim.Adam(enhancemodel.parameters(), lr=lr_start, weight_decay=0.0001)
    # load LFN
    if num>0:
        fusionmodel = model.FusionNet()
        LFN_model = luminance_feedback()
        if logger == None:
            logger = logging.getLogger()
            setup_logger(modelpth)
        LFN_model.load_state_dict(torch.load(osp.join(os.getcwd(),'checkpoint/best_cls.pth')),False)
        fusionmodel_path = osp.join(os.getcwd(),'model',str(num), 'fusion_model.pth')
        fusionmodel.load_state_dict(torch.load(fusionmodel_path),False)
        LFN_model=LFN_model.cuda()
        fusionmodel=fusionmodel.cuda()
        LFN_model.eval()
        fusionmodel.eval()
        for p in LFN_model.parameters():
            p.requires_grad = False
        for q in fusionmodel.parameters():
            q.requires_grad = False
        # Freeze LFN network parameters

    datas = dataloader.enhance_dataset_loader('train') # load training data
    datal = torch.utils.data.DataLoader(datas, batch_size=enhance_batch_size,shuffle=True,num_workers=n_workers,pin_memory=False)
    print("the training dataset is length:{}".format(datas.length))
    datal.n_iter = len(datal)
    
    # fusion_loss
    L_color = enhancement_loss.L_color()
    L_spa = enhancement_loss.L_spa()
    L_exp = enhancement_loss.L_exp(8,0.5)
    L_TV = enhancement_loss.L_TV()
    grad_acc_steps=4.0 # gradient accumulation to increase batch_size
    epoch=20
    st = glob_st = time.time()
    logger.info('Training Enhancement Model start~')
    for epo in range(0, epoch):
        lr_decay=0.75
        lr_this_epo=lr_start*lr_decay**((epo/5)+1)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_this_epo
        for it,(image_vis,image_ir) in enumerate(datal):
                enhancemodel.train()
                image_vis = Variable(image_vis,requires_grad=True).cuda()
                image_ir = Variable(image_ir,requires_grad=True).cuda()
                enhanced_image_1,enhanced_image,A  = enhancemodel(image_vis)
                loss_TV = 200*L_TV(A)
                loss_spa = torch.mean(L_spa(enhanced_image, image_vis))
                loss_col = 5*torch.mean(L_color(enhanced_image))
                loss_exp = 10*torch.mean(L_exp(enhanced_image))
                loss_enhance =  loss_TV + loss_spa + loss_col + loss_exp
                # 亮度反馈损失
                if num>0:
                    image_vis_en_ycbcr = rgb2ycbcr(enhanced_image)
                    vi_clahe_y=dataloader.clahe(image_vis_en_ycbcr[:,0:1,:,:],image_vis_en_ycbcr.shape[0])
                    _,_,_,Y_f = fusionmodel(vi_clahe_y,image_ir)
                    fusion_ycbcr = torch.cat((Y_f,image_vis_en_ycbcr[:,1:2,:,:],image_vis_en_ycbcr[:,2:,:,:]),dim=1)
                    fusion_image_if = ycbcr2rgb(fusion_ycbcr)
                    ones = torch.ones_like(fusion_image_if)
                    zeros = torch.zeros_like(fusion_image_if)
                    fusion_image_if = torch.where(fusion_image_if > ones, ones, fusion_image_if)
                    fusion_image_if = torch.where(fusion_image_if < zeros, zeros, fusion_image_if)
                    pred=LFN_model(fusion_image_if)
                    day_p=pred[:,0]
                    night_p=pred[:,1]
                    w_day=day_p/(day_p+night_p)
                    illu_loss=0#初始化
                    illu_loss = math.fabs(-1*math.log((torch.mean(w_day)+1E-6),1.5))#亮度反馈损失函数
                    loss_total = loss_enhance + 0.1*(num) * illu_loss#迭代变更权重
                else:
                    loss_total = loss_enhance
                
                if grad_acc_steps>1:
                    loss_total=loss_total/grad_acc_steps
                loss_total.backward()
                ed = time.time()
                t_intv, glob_t_intv = ed - st, ed - glob_st
                now_it = datal.n_iter * epo + it + 1
                eta = int((datal.n_iter * epoch - now_it)* (glob_t_intv / (now_it)))
                eta = str(datetime.timedelta(seconds=eta))
                if now_it%grad_acc_steps==0:
                    optimizer.step()
                    optimizer.zero_grad()
                if now_it % 50 == 0:
                    if num>0:
                        loss_illu=illu_loss
                    else:
                        loss_illu=0
                    msg = ', '.join(
                        [
                            'step: {it}/{max_it}',
                            'loss_total: {loss_total:.4f}',
                            'loss_enhance: {loss_enhance:.4f}',
                            'loss_illu: {loss_illu:.4f}',
                            'loss_TV: {loss_TV:.4f}',
                            'loss_spa: {loss_spa:.6f}',
                            'loss_col: {loss_col:.4f}',
                            'loss_exp: {loss_exp:.4f}',
                            'eta: {eta}',
                            'time: {time:.4f}',
                        ]
                    ).format(
                        it=now_it,
                        max_it=datal.n_iter * epoch,
                        loss_total=loss_total.item(),
                        loss_enhance=loss_enhance.item(),
                        loss_illu=loss_illu,
                        loss_TV=loss_TV.item(),
                        loss_spa=loss_spa.item(),
                        loss_col=loss_col.item(),
                        loss_exp=loss_exp.item(),
                        time=t_intv,
                        eta=eta,)
                    logger.info(msg)
                    st = ed
    enhance_model_file = osp.join(modelpth, 'enhancement_model.pth')
    torch.save(enhancemodel.state_dict(), enhance_model_file)
    logger.info("Enhancement Model Save to: {}".format(enhance_model_file))
    logger.info('\n')
    
def run_enhance(i): # LAN eval
    enhance_model_path = osp.join(os.getcwd(),'model',str(i+1), 'enhancement_model.pth')
    enhanced_dir=osp.join(os.getcwd(),'eval/vi_en')
    os.makedirs(enhanced_dir, mode=0o777, exist_ok=True)
    enhancemodel = model.luminance_adjustment()
    enhancemodel = enhancemodel.cuda()
    enhancemodel.eval()
    enhancemodel.load_state_dict(torch.load(enhance_model_path))
    print('enhancemodel,done!')
    test_dataset = dataloader.enhance_dataset_loader_test(osp.join(os.getcwd(),'eval'))
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        pin_memory=False,
        drop_last=False,
    )
    test_tqdm=tqdm(test_loader,total=len(test_loader))
    with torch.no_grad():
        for images_vis, name in test_tqdm:
            images_vis = Variable(images_vis)
            images_vis = images_vis.cuda()
            _,enhanced_image,_ = enhancemodel(images_vis)
            enhanced_image = enhanced_image.cpu().numpy() #转为array
            for k in range(len(name)):
                image = enhanced_image[k, :, :, :] #丢掉第0维
                image = image.squeeze()
                image=torch.tensor(image).to(images_vis.device)
                image=transforms.ToPILImage()(image)

                save_path = osp.join(enhanced_dir,str(i+1),name[k])
                image.save(save_path)

def run_fusion(i): #RFN eval
    fusion_model_path = osp.join(os.getcwd(),'model',str(i+1), 'fusion_model.pth')
    fusion_dir=osp.join(os.getcwd(),'eval')
    os.makedirs(fusion_dir, mode=0o777, exist_ok=True)
    fusionmodel = model.FusionNet()
    fusionmodel = fusionmodel.cuda()
    fusionmodel.eval()
    fusionmodel.load_state_dict(torch.load(fusion_model_path))
    print('fusionmodel,done!')
    testdataset = dataloader.fusion_dataset_loader_eval(i,osp.join(os.getcwd(),'eval'))
    testloader = DataLoader(
        dataset=testdataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
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

            image_vis_en_ycbcr = rgb2ycbcr(images_vis)
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

                save_path1 = osp.join(fusion_dir,'If',str(i+1), name[k])
                image_I_f.save(save_path1) 

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train with pytorch')
    # parser.add_argument('--batch_size', '-B', type=int, default=8)
    parser.add_argument('--num_workers', '-j', type=int, default=8)
    args = parser.parse_args()
    logpath='./logs'
    logger = logging.getLogger()
    setup_logger(logpath)
    for i in range (0,5):
        train_enhancement(i, logger)  # LAN train
        print("|{0} Train LAN Sucessfully~!".format(i + 1))
        run_enhance(i)  # LAN eval
        print("|{0} Enhancement Image Sucessfully~!".format(i + 1))
        train_fusion(i, logger)  # RFN train
        print("|{0} Train Fusion Model Sucessfully~!".format(i + 1))
        run_fusion(i)  # RFN eval
        print("|啊哈哈|{0} Fusion Image Sucessfully~!".format(i + 1))
        print("———————————————————————————")
    print("Training Done!")
