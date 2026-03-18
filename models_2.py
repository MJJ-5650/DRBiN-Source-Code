import torch
import torch.nn as nn
import torch.nn.functional as F
import mmcv
from mmcv.cnn import ConvModule
import numpy as np
from model.non_local import Network
from model.builder import build_backbone, build_head, build_neck
from mmcv.cnn import build_conv_layer, build_norm_layer

from model.resnet import ResLayer, BasicBlock
from model.backbone import Backbone
from model.backbone import Neck
from model.backbone import Head
class MC_withoutRank_v2(nn.Module):
    def __init__(self, H, W, rank,
            backbone = dict(
                type='ResNet',
                depth=8,
                in_channels=1,  # input channel
                num_stages=4,  # the output stage
                out_indices=[0, 1, 2, 3],  # the indices
                dilations=(1, 1, 1, 1),  # (1, 1, 2, 4),
                strides=(1, 2, 2, 2),  # 1, 2, 1, 1),
                norm_eval=False,
                norm_cfg=dict(type='BN', requires_grad=True),
                # init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet18')
                ),
            neck = dict(
                type='FPN',
                in_channels=[1, 64, 128, 256],  # in_channels=[32, 64, 64, 128, 256, 512]
                out_channels=64,
                num_outs=3,
                # upsample_cfg=dict(mode='bilinear'),
                # norm_cfg=dict(type='BN', requires_grad=True),
                # act_cfg=dict(type='ReLU'),            
                ),
            head = dict(
                type='FPNHead',
                # in_channels=[128, 128, 128, 128, 128, 128],
                # in_index= [0, 1, 2, 3, 4, 5],
                # feature_strides= [4, 8, 16, 32, 64, 128],
                in_channels=[64, 64, 64], # in_channels=[64, 64, 64,  64, 64],
                in_index=[0, 1, 2], # in_index=[0, 1, 2, 3, 4],
                feature_strides=[4, 8, 16], # feature_strides=[4, 8, 16, 32, 64],
                channels=32,
                # input_transform=,
                dropout_ratio=0,
                out_channels=1,
                norm_cfg=dict(type='BN', requires_grad=True),
                # act_cfg=dict(type='ReLU'),
                align_corners=None,
                loss_decode=dict(
                    type='MSELoss', loss_weight=1.0,reduction='mean')
                )
            ):
        super(MC_withoutRank_v2, self).__init__()

        # build backbone
        # self.backbone = build_backbone(backbone)

        # # init the learnable param
        # self.backbone.init_weights()
        self.backbone = Backbone()
        self.neck = Neck()
        self.loss_r = nn.L1Loss(reduction='sum')
        # self.neck = build_neck(neck)
        
        self.head = build_head(head) # FPNHead(in_channels, )
        self.H = H
        self.W = W
        self.rank = rank
        Hy = 50

        self.branch_u_r = nn.Sequential(
            torch.nn.Linear(W, W, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(W, rank, bias=False),
            torch.nn.ReLU()
        )

        self.branch_v_r = nn.Sequential(
            torch.nn.Linear(H, H, bias=False),
            torch.nn.Tanh(),
            torch.nn.Linear(H, rank, bias=False),
            torch.nn.ReLU()
        )

        self.branch_r = nn.Sequential(
            torch.nn.Linear(rank, rank, bias=False),
            torch.nn.ReLU(),
            # torch.nn.Linear(rank//2, rank, bias=False),
            # torch.nn.ReLU()
        )

    def forward(self, index, x1, x2, x3, img, epoch, noise):  # input是输入的三个矩阵
        U = self.branch_u_r(x1)
        V = self.branch_v_r(x2)
        r = self.branch_r(x3)

        R = r.repeat(U.shape[0], 1)
        U_R = torch.mul(U,R)
        V_R = torch.mul(V,R)

        out_rank = torch.mm(U_R, V_R.t())[None]

        out = torch.sigmoid(out_rank)

        backbone = self.backbone(out[None])
        backbone = list(backbone)
        # backbone.insert(0, out[None])

        neck = self.neck(backbone)

        losses = self.head.forward_train(neck, out, img[None], index, out_features=None)#1

        loss_R = self.loss_r(r, torch.zeros(self.rank).type_as(R))

        # weight = (losses['MSELoss']/loss_R).detach().cpu()
        losses['MSELoss'] +=2e-6*loss_R
        return losses, out

    def forward_test(self, index, x1, x2, x3, img, noise):  # input是输入的三个矩阵
        U = self.branch_u_r(x1)
        V = self.branch_v_r(x2)
        r = self.branch_r(x3)

        R = r.repeat(U.shape[0], 1)
        U_R = torch.mul(U,R)
        V_R = torch.mul(V,R)

        out_rank = torch.mm(U_R, V_R.t())[None]
        
        out = torch.sigmoid(out_rank)
        backbone = self.backbone(out[None])
        backbone = list(backbone)
        # backbone.insert(0, out[None])
        
        neck = self.neck(backbone)

        pred, pred_rank = self.head.forward_test(neck, out, img[None], out_features=None)

        return pred, pred_rank

