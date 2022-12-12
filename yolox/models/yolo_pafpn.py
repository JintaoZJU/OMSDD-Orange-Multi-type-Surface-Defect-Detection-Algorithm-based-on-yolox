#!/usr/bin/env python
# -*- encoding: utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.

import torch
import torch.nn as nn

from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv, SE, CBAM, CAM


class YOLOPAFPN(nn.Module):
    """
    YOLOv3 model. Darknet 53 is the default backbone of this model.
    """

    def __init__(
        self,
        depth=1.0,
        width=1.0,
        in_features=("dark3", "dark4", "dark5"),
        in_channels=[256, 512, 1024],
        depthwise=False,
        act="silu",
    ):
        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise, act=act)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(
            int(in_channels[2] * width), int(in_channels[1] * width), 1, 1, act=act #dark5 1*1conv 降维 cin=1024 cout=512
        )
        self.C3_p4 = CSPLayer(
            int(2 * in_channels[1] * width), #dark4 & lateral_conv0 concat cin=1024 cout=512
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )  # cat

        self.reduce_conv1 = BaseConv(
            int(in_channels[1] * width), int(in_channels[0] * width), 1, 1, act=act
        )
        self.C3_p3 = CSPLayer(
            int(2 * in_channels[0] * width), #dark3 & 上一个top-down特征图 concat
            int(in_channels[0] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv2 = Conv(
            int(in_channels[0] * width), int(in_channels[0] * width), 3, 2, act=act
        )
        self.C3_n3 = CSPLayer(
            int(2 * in_channels[0] * width),
            int(in_channels[1] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        # bottom-up conv
        self.bu_conv1 = Conv(
            int(in_channels[1] * width), int(in_channels[1] * width), 3, 2, act=act
        )
        self.C3_n4 = CSPLayer(
            int(2 * in_channels[1] * width),
            int(in_channels[2] * width),
            round(3 * depth),
            False,
            depthwise=depthwise,
            act=act,
        )

        ###### 实例化Attention对象 ######        
        '''in_features=("dark3", "dark4", "dark5")
           in_channels=[256, 512, 1024]'''
        ##### SE ######  
        self.se2 = SE(int(in_channels[2]*width)) #dark5+SE C=1024
        self.se1 = SE(int(in_channels[1]*width)) #dark4+SE C=512
        self.se0 = SE(int(in_channels[0]*width)) #dark3+SE C=256

        ##### CBAM ######
        self.CBAM2 = CBAM(int(in_channels[2]*width)) #dark5+CBAM C=1024
        self.CBAM1 = CBAM(int(in_channels[1]*width)) #dark4+CBAM C=512
        self.CBAM0 = CBAM(int(in_channels[0]*width)) #dark3+CBAM C=256

        ###### CAM ######
        self.CAM2 = CAM(int(in_channels[2]*width)) #dark5+CAM C=1024
        self.CAM1 = CAM(int(in_channels[1]*width)) #dark4+CAM C=512
        self.CAM0 = CAM(int(in_channels[0]*width)) #dark3+CAM C=256
        ################################################################

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
        """

        #  backbone x2, x1, x0 --> 256 512 1024
        out_features = self.backbone(input)
        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features

        ################# 接入Attention 开始#################
        ###### SE #####
        # x2 = self.se0(x2)
        # x1 = self.se1(x1)
        # x0 = self.se2(x0)

        ###### CBAM #####
        # x2 = self.CBAM0(x2)
        # x1 = self.CBAM1(x1)
        # x0 = self.CBAM2(x0)
        
        ###### CAM #####
        # x2 = self.CAM0(x2)
        # x1 = self.CAM1(x1)
        # x0 = self.CAM2(x0)

        ################# 接入Attention 结束 #################

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

                      #  backbone x2, x1, x0 --> 256 512 1024
                      #################  Bi-FPN 开始 n=2 #################
         #################  增加 x2, x1, x0 到pan_out的shortcut 并重复模块一次 
        x2 = x2 + pan_out2
        x1 = x1 + pan_out1
        x0 = x0 + pan_out0

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        #################  增加 x2, x1, x0 到pan_out的shortcut 并重复模块一次 
        x2 = x2 + pan_out2
        x1 = x1 + pan_out1
        x0 = x0 + pan_out0

        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = torch.cat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = torch.cat([f_out1, x2], 1)  # 256->512/8
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8

        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = torch.cat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = torch.cat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        pan_out2 = x2 + pan_out2
        pan_out1 = x1 + pan_out1
        pan_out0 = x0 + pan_out0
                #################  Bi-FPN 结束  #################

                ################# 后接入Attention #################
        ###### SE #####
        # pan_out2 = self.se0(pan_out2)
        # pan_out1 = self.se1(pan_out1)
        # pan_out0 = self.se2(pan_out0)

        ###### CBAM #####
        # pan_out2 = self.CBAM0(pan_out2)
        # pan_out1 = self.CBAM1(pan_out1)
        # pan_out0 = self.CBAM2(pan_out0)
        
        ###### CAM #####
        # pan_out2 = self.CAM0(pan_out2)
        # pan_out1 = self.CAM1(pan_out1)
        # pan_out0 = self.CAM2(pan_out0)

                ################# 接入Attention  #################

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
