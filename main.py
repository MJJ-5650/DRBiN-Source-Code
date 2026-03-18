import time
import cv2
import random
import os
from utils.logger import Logger
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import models_2 as models
import mmcv
import scipy.io as io
from skimage import util
from torch.utils.tensorboard import SummaryWriter
from util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
import cProfile

def to_tensor(data):
    """Convert objects of various python types to :obj:`torch.Tensor`.

    Supported types are: :class:`numpy.ndarray`, :class:`torch.Tensor`,
    :class:`Sequence`, :class:`int` and :class:`float`.
    """
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, Sequence) and not mmcv.is_str(data):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError(
            f'Type {type(data)} cannot be converted to tensor.'
            'Supported types are: `numpy.ndarray`, `torch.Tensor`, '
            '`Sequence`, `int` and `float`')


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model')
    # parser.add_argument('config', help='train config file path')
    # parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument(
        '--input_image', default='./data/Images', help='the input image')

    parser.add_argument(
        '--output', default='./MAI/Main', help='the output image')  
    args = parser.parse_args()

    return args


# @profile
def main(args):
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    
    # MAI para
    rank_arr = []

    seed = 0
    obsratio_list = np.arange(1,11)/10
    obsratio_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    variance_list = [0, 0.002, 0.005]
    img_list = np.arange(1, 17)

    # For Test
    obsratio_list= [0.5]
    variance_list = [0]
    img_list = [12]
    lr_rate_list = [0.002]
    rak_list = [400]
    # img_list = [2]

    for obsratio in obsratio_list:
        for variance in variance_list:
            # data pipeline
            # 1. load image
            for img_ind in img_list:
                file_name= str(img_ind)+'.jpg'
                if seed is not None:
                    random.seed(seed)
                    # os.environ['PYTHONHASHSEED'] = str(seed)
                    np.random.seed(seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed(seed)
                        torch.cuda.manual_seed_all(seed)
                    torch.manual_seed(seed)
                    torch.backends.cudnn.enabled = True
                    torch.backends.cudnn.deterministic = True
                    torch.backends.cudnn.benchmark = False
                # if seed == 0:
                #     torch.backends.cudnn.deterministic = True
                #     torch.backends.cudnn.benchmark = False

                # 0. set hyper-param
                for rak in rak_list:
                    for lr_rate in lr_rate_list: 
                        weight_decay_rate, num_iter, step = 0.0, 1000, 200
                        
                        backbone = dict(
                            type='ResNet',
                            # base_channels=32,
                            depth=8,
                            in_channels=1,  # input channel
                            num_stages=4,  # the output stage
                            out_indices=[0, 1, 2, 3],  # the indices
                            dilations=(1, 1, 1, 1),  # (1, 1, 2, 4),
                            strides=(1, 2, 2, 2),  # 1, 2, 1, 1),
                            norm_eval=False,
                            norm_cfg=dict(type='BN', requires_grad=True),
                            # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
                            )
                        neck = dict(
                            type='FPN',
                            in_channels=[64, 128, 256, 512],  # in_channels=[32, 64, 64, 128, 256, 512]
                            out_channels=64,
                            num_outs=4,
                            upsample_cfg=dict(mode='bilinear'), # bilinear nearest
                            # norm_cfg=dict(type='BN', requires_grad=True),
                            # act_cfg=dict(type='LeakyReLU'),            
                            )

                        head = dict(
                            type='FPNHead',
                            # in_channels=[128, 128, 128, 128, 128, 128],
                            # in_index= [0, 1, 2, 3, 4, 5],
                            # feature_strides= [4, 8, 16, 32, 64, 128],
                            in_channels=[64, 64, 64, 64], # in_channels=[64, 64, 64,  64, 64],
                            in_index=[0, 1, 2, 3], # in_index=[0, 1, 2, 3, 4],
                            feature_strides=[1, 2, 4, 8], # feature_strides=[4, 8, 16, 32, 64],
                            channels=32,
                            # input_transform=,
                            dropout_ratio=0.0000,
                            out_channels=1,
                            norm_cfg=dict(type='BN', requires_grad=True), # cannot be removed
                            # act_cfg=dict(type='ReLU'),
                            align_corners=None,
                            loss_decode=dict(
                                type='MSELoss', loss_weight=1.0,reduction='mean')
                            )
                        # init output path
                        exp_i = file_name.split('.jpg')[0]
                        output_path = os.path.join(args.output, exp_i, str(obsratio)+'_'+str(variance))
                        if not os.path.exists(output_path):
                            os.makedirs(output_path)
                        
                        # init logc
                        log_path = os.path.join(output_path, str(obsratio)+'_'+str(rak)+'_'+str(lr_rate)+'.log')
                        logger = Logger(log_path)
                        logger = Logger(log_path, True)
                        
                        # add hyper-param to the log
                        logger.append(f"seed: {seed}, deterministic: {torch.backends.cudnn.deterministic}, benchmark: {torch.backends.cudnn.benchmark}")
                        if seed is not None:
                            logger.append(f'random: {np.random.rand(500, 500)[0,:20]}')
                        logger.append(f"obsratio: {obsratio}, rank: {rak}, lr_rate: {lr_rate}, step: {step}, weight_decay: {weight_decay_rate}, num_iter: {num_iter}") 
                        logger.append(f"backbone_cfg: {backbone}")
                        logger.append(f"neck_cfg: {neck}")
                        logger.append(f"head_cfg: {head}")
                        # 1. load data
                        file_path = os.path.join(args.input_image, file_name)
                        logger.append(f'Image_name:{file_path}')
                        img_ori = cv2.imread(file_path)
                        img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)[..., None]
                        # img_ori = cv2.resize(img_ori, [224, 224])[...,None]
                        # 2. add noise
                        img = util.random_noise(img_ori.copy(), mode='gaussian', seed=seed, var=variance) * 255  # 0.014
                        # img = img_ori + np.random.randn(*img_ori.shape) * 0
                        # img = img_ori.copy()
                        # 3. incomplete
                        H, W, _ = img.shape
                        np.random.seed(seed)
                        random_value = np.random.rand(H, W)
                        index = np.where(random_value > obsratio)
                        index_non = np.where(random_value < obsratio)
                        ind_len = len(index_non[0])
                        
                        index_original = index
                        img[index[0][:], index[1][:], :] = 0
                        img_mask = torch.ones([H, W]).to(device)
                        img_mask[index[0][:], index[1][:]] = 0
                        img = img / 255.
                        img = np.clip(img, 0, 1).astype(np.float32)
                        
                        cv2.imwrite(f'{output_path}/input.png', img*255)
                        img = to_tensor(img.transpose(2, 0, 1)).float().to(device)  # CHW

                        x1, x2, x3 = torch.eye(H).to(device), torch.eye(W).to(device), torch.ones(rak).to(device)

                        # for round_i in range(args.round):
                        #     logger.append(f"round_{round_i}")
                        model = models.MC_withoutRank_v2(H, W, rak, backbone, neck, head).to(device)

                        opt = optim.Adam(model.parameters(), lr=lr_rate, weight_decay=weight_decay_rate)

                        start_train = time.time()
                        writer = SummaryWriter(output_path)
                        
                        img_train = img.clone()
                        for epoch in range(num_iter):
                            
                            # if random.random() < 0.5:
                            #     random_ind = random.sample(range(ind_len), ind_len//2)
                            #     img_train[0,index_non[0][random_ind], index_non[1][random_ind]] = 0
                            # elif random.random() < 0.2:
                            #     random_ind = random.sample(range(ind_len), ind_len//5)
                            #     img_train[0,index_non[0][random_ind], index_non[1][random_ind]] = 0
                            # elif random.random() < 0.6:
                            #     random_ind = random.sample(range(ind_len), ind_len//10)
                            #     img_train[0,index_non[0][random_ind], index_non[1][random_ind]] = 0
                            # elif random.random() < 0.7:
                            #     random_ind = random.sample(range(ind_len), ind_len//20)
                            #     img_train[0,index_non[0][random_ind], index_non[1][random_ind]] = 0
                            
                            model.train()
                            # img_0 = (img.clone()*255 + torch.normal(0, 0.0001, img.shape).type_as(img))/255
                            noise = img.detach().clone().normal_() * 0.005
                            losses, mid_image = model(index, x1, x2, x3, img, epoch, noise=noise) # MAI: mid_image for Fig
                            if epoch % 100 == 0:
                                mid_image = mid_image.detach().cpu().numpy()
                                cv2.imwrite(f'{output_path}/Mid_Image_{epoch}_{obsratio}_{variance}.png', np.squeeze(mid_image*255))
                            
                            if epoch == step:
                                opt.param_groups[0]['lr'] = 0.001
                            opt.zero_grad()
                            mse_loss = losses['MSELoss']
                            # loss_rank = losses['low_rank']
                            loss = mse_loss
                            writer.add_scalar('Loss/train', loss.item(), epoch)

                            loss.backward()
                            # torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=30, norm_type=2)
                            opt.step()

                            # MAI: rank curve gen
                            # with torch.no_grad():
                            #     mid_mai = mid_image.cpu().numpy()
                            #     mid_mai = Image.fromarray(np.squeeze(mid_mai*255))
                            #     mid_rank = np.linalg.matrix_rank(mid_mai)
                            #     rank_arr.append(mid_rank)
                            #     if epoch == num_iter - 1:
                            #         np.save('MAI/rank_arr.npy', rank_arr)

                            if epoch % 1 == 0 and epoch != 0:
                                with torch.no_grad():
                                    log_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                                    model.eval()
                                    output, out_rank = model.forward_test(index_original, x1, x2, x3, img, noise=noise) # output_shape = (1,1,500,500)
                                    output = output[0].transpose(1, 2, 0) # output_shape = (500,500,1)
                                    # out_rank = out_rank[0].transpose(1,2,0)
                                    output = output * 255.
                                    # out_rank = out_rank * 255.
                                    psnr = compare_psnr(output.squeeze()/255., np.array(img_ori.squeeze())/255.)
                                    ssim = calculate_ssim(output.squeeze(), np.array(img_ori.squeeze()), 0)

                                    # FlowChat Setting
                                    # if epoch % 100 == 0:
                                    #     flowchart_img_path = 'MAI/MAI_FLowChart_Imge/Var_'+str(variance)+'_Obser_'+ str(obsratio)
                                    #     if not os.path.exists(flowchart_img_path):
                                    #         os.makedirs(flowchart_img_path)
                                    #     out_mai = Image.fromarray(np.squeeze(output))
                                    #     out_mai=out_mai.convert('L')
                                    #     out_mai.save(flowchart_img_path + '/Img'+str(img_ind)+'_out_'+str(epoch)+'_psnr_'+
                                    #                 str(round(psnr,2))+'_ssim_'+str(round(ssim,2))+'.png')
                                    
                                    #     # mid_mai = mid_image.cpu().numpy()
                                    #     mid_mai = mid_mai .convert('L')
                                    #     mid_mai.save(flowchart_img_path + '/Img'+str(img_ind)+'_mid_'+str(epoch)+'_psnr_'+
                                    #                 str(round(psnr,2))+'_ssim_'+str(round(ssim,2))+'.png')

                                    # if epoch%100 ==0 and epoch !=0:
                                    #     cv2.imwrite(f'{output_path}/{exp_i}_{epoch}_{obsratio}_{variance}_{round(psnr,2)}_{round(ssim,2)}.png', output)
                                    
                                    # psnr_rank = calculate_psnr(out_rank.squeeze()/255,np.array(img_ori.squeeze()/255),0)
                                    writer.add_scalar('Accuracy/PSNR', psnr, epoch)
                                    # writer.add_scalar('Accuracy/PSNR_rank', psnr_rank, epoch)
                                    # print(f"epoch: {epoch}, psnr: {psnr}")
                                    
                                    res_str = f"epoch: {epoch}, psnr: {psnr}, ssim: {ssim}"
                                    logger.append(log_str + ': ' + res_str + f", lr_rate: {opt.param_groups[0]['lr']}")

                        end_train = time.time()
                        # print(f"runtime: {end_train - start_train}")
                        logger.append(f"runtime: {end_train - start_train}")
                        
                        save_path= os.path.join(output_path, str(obsratio)+'_'+str(num_iter)+'_.mat')
                        # '14_'+ str(obsratio)+str(num_iter)+'.mat'
                        io.savemat(save_path,
                                    {'I_result': output.squeeze() / 255,
                                    'ob_ratio': obsratio,
                                    "var": variance,
                                    'num_iter': num_iter, 'lr_rate': lr_rate,
                                    'TrainTime': end_train - start_train})

if __name__ == '__main__':
    args = parse_args()
    main(args)
    # cProfile.run('main(args)')