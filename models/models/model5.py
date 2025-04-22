import pretrainedmodels
import torch
import torch.nn as nn
import numpy as np
from math import sqrt
import math
from PIL import Image
from torchvision import transforms
from torchvision import transforms as trans
from torch.nn import functional as F
# from Model_Prepare.Models.xception import xception
from pytorch_wavelets import DWTForward, DWTInverse
from pretrainedmodels import xception
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'




class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=8):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.sharedMLP = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x):
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd"
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 计算平均池和最大池输出
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        # 合并平均池和最大池的结果
        combined = torch.cat([avg_out, max_out], dim=1)

        # 通过卷积层获得注意力图
        attention = self.conv1(combined)
        attention = self.sigmoid(attention)

        # 将注意力图广播到原始输入的每个通道上并应用
        return x * attention.expand_as(x)


class DualCrossModalAttention(nn.Module):
    """ Dual CMA attention Layer"""

    def __init__(self, in_dim, activation=None, size=14, ratio=8, ret_att=False):  # size=16
        super(DualCrossModalAttention, self).__init__()
        self.chanel_in = in_dim
        self.activation = activation
        self.ret_att = ret_att

        # query conv
        self.key_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim // ratio, kernel_size=1)
        self.key_conv_share = nn.Conv2d(
            in_channels=in_dim // ratio, out_channels=in_dim // ratio, kernel_size=1)
        # self.adjust_conv = nn.Conv2d(in_channels=8, out_channels=91, kernel_size=1)  # 降维卷积

        self.linear1 = nn.Linear(size * size, size * size)
        self.linear2 = nn.Linear(size * size, size * size)

        # separated value conv
        self.value_conv1 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma1 = nn.Parameter(torch.zeros(1))

        self.value_conv2 = nn.Conv2d(
            in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data, gain=0.02)

    def forward(self, x, y):
        """
            inputs :
                x : input feature maps( B X C X W X H)
            returns :
                out : self attention value + input feature
                attention: B X N X N (N is Width*Height)
        """
        x = F.interpolate(x, size=(14, 14), mode='bilinear', align_corners=False)

        B, C, H, W = x.size()  # 4,728,14,14


        def _get_att(a, b):
            size = 14  # 选择较小的尺寸以减少计算量，或根据需要选择其他尺寸
            a = F.interpolate(a, size=(size, size), mode='bilinear', align_corners=False)
            b = F.interpolate(b, size=(size, size), mode='bilinear', align_corners=False)

            proj_key1 = self.key_conv_share(self.key_conv1(a)).view(
                B, -1, H * W).permute(0, 2, 1)  # B, HW, C
            proj_key2 = self.key_conv_share(self.key_conv2(b)).view(
                B, -1, H * W)  # B X C x (*W*H)

            energy = torch.bmm(proj_key1, proj_key2)  # B, HW, HW  4

            attention1 = self.softmax(self.linear1(energy))
            attention2 = self.softmax(self.linear2(
                energy.permute(0, 2, 1)))  # BX (N) X (N)

            return attention1, attention2

        att_y_on_x, att_x_on_y = _get_att(x, y)
        # proj_value_y_on_x = self.value_conv2(y).view(
        #     B, -1, H * W)  # B, C, HW
        proj_value_y_on_x = self.value_conv2(y)
        proj_value_y_on_x = F.interpolate(proj_value_y_on_x, size=(14, 14), mode='bilinear', align_corners=False)

        proj_value_y_on_x = proj_value_y_on_x.view(B, -1, H * W)
        out_y_on_x = torch.bmm(proj_value_y_on_x, att_y_on_x.permute(0, 2, 1))
        out_y_on_x = out_y_on_x.view(B, C, H, W)
        out_x = self.gamma1 * out_y_on_x + x

        proj_value_x_on_y = self.value_conv1(x).view(
            B, -1, H * W)  # B , C , HW
        out_x_on_y = torch.bmm(proj_value_x_on_y, att_x_on_y.permute(0, 2, 1))
        out_x_on_y = out_x_on_y.view(B, C, H, W)
        y = F.interpolate(y, size=(14, 14), mode='bilinear', align_corners=False)

        out_y = self.gamma2 * out_x_on_y + y

        if self.ret_att:
            return out_x, out_y, att_y_on_x, att_x_on_y

        return out_x, out_y  # , attention


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan=2048 * 2, out_chan=2048, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = nn.Sequential(
            nn.Conv2d(in_chan, out_chan, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_chan),
            nn.ReLU()
        )
        self.ca = ChannelAttention(out_chan, ratio=16)
        self.init_weight()

    def forward(self, x, y):
        fuse_fea = self.convblk(torch.cat((x, y), dim=1))
        fuse_fea = fuse_fea + fuse_fea * self.ca(fuse_fea)
        return fuse_fea

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None:
                    nn.init.constant_(ly.bias, 0)


class HiFE(nn.Module):

    def __init__(self, J=1, num_classes=2):
        super(HiFE, self).__init__()
        self.xception = pretrainedmodels.xception(pretrained='imagenet')
        self.xception.last_linear = nn.Linear(2048, num_classes)
        self.channel_downsample = nn.Conv2d(2048, 728, 1)  # 降低通道数匹配Attention需要的728
        self.binary_classifier = nn.Linear(728, num_classes)
        self.channel_attention = ChannelAttention(in_planes=2048)
        self.spatial_attention = SpatialAttention()

        self.xfm1 = DWTForward(J=J, mode='zero', wave='haar')
        self.ifm1 = DWTInverse(mode='zero', wave='haar')

        self.xfm2 = DWTForward(J=J, mode='zero', wave='haar')
        self.ifm2 = DWTInverse(mode='zero', wave='haar')

        self.xfm3 = DWTForward(J=J, mode='zero', wave='haar')
        self.ifm3 = DWTInverse(mode='zero', wave='haar')

        self.xfm4 = DWTForward(J=J, mode='zero', wave='haar')
        self.ifm4 = DWTInverse(mode='zero', wave='haar')

        self.dual_cma1 = DualCrossModalAttention(in_dim=728, ret_att=False)
        self.fusion1 = FeatureFusionModule(in_chan=728 * 2, out_chan=728)

        self.conv_x1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x4 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1, groups=3),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )

        self.conv_x1_2_1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_2_2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_2_3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_3_1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_3_2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_3_3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_4_1 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_4_2 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x1_4_3 = nn.Sequential(
            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0, groups=9),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),
        )
        self.conv_x2_1 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_x2_2 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )

        self.conv_x2_3 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_x3_1 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_x3_2 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_x3_3 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_x4_1 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_x4_2 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_x4_3 = nn.Sequential(
            nn.Conv2d(12, 9, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(),

            nn.Conv2d(9, 9, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(9),
            nn.ReLU(True),

            nn.Conv2d(9, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_xl = nn.Sequential(
            nn.Conv2d(3, 3, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(3),
            nn.ReLU(),

            nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 6, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(6),
            nn.ReLU(),

            nn.Conv2d(6, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
        )
        self.conv_xh = nn.Sequential(
            nn.Conv2d(9, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.ReLU(),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.Conv2d(256, 728, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
        )
        self.bn = nn.BatchNorm2d(3)

    def extract_features(self, x):

        # one
        Yl_1, Yh_1 = self.xfm1(x)
        # x1_1 = Yh_1[0][:, :, 0, :, :].view(x.size(0), -1, 112, 112)
        # x1_2 = Yh_1[0][:, :, 1, :, :].view(x.size(0), -1, 112, 112)
        # x1_3 = Yh_1[0][:, :, 2, :, :].view(x.size(0), -1, 112, 112)
        x1_1 = F.interpolate(Yh_1[0][:, :, 0, :, :], size=(112, 112), mode='bilinear', align_corners=False)
        x1_2 = F.interpolate(Yh_1[0][:, :, 1, :, :], size=(112, 112), mode='bilinear', align_corners=False)
        x1_3 = F.interpolate(Yh_1[0][:, :, 2, :, :], size=(112, 112), mode='bilinear', align_corners=False)
        x1 = torch.cat((x1_1, x1_2, x1_3), dim=1)
        x1 = self.conv_x1(x1)

        x1_1 = x1[:, 0: 3, :, :]
        x1_2 = x1[:, 3: 6, :, :]
        x1_3 = x1[:, 6: 9, :, :]

        # two
        yl_2, yh_2 = self.xfm2(x1_1)
        yl_3, yh_3 = self.xfm2(x1_2)
        yl_4, yh_4 = self.xfm2(x1_3)

        x1_2_1 = torch.cat((yh_2[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56),
                            yh_3[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56),
                            yh_4[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56)), dim=1)
        x1_2_2 = torch.cat((yh_2[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56),
                            yh_3[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56),
                            yh_4[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56)), dim=1)
        x1_2_3 = torch.cat((yh_2[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56),
                            yh_3[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56),
                            yh_4[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56)), dim=1)

        x1_2_1 = self.conv_x1_2_1(x1_2_1)
        x1_2_2 = self.conv_x1_2_2(x1_2_2)
        x1_2_3 = self.conv_x1_2_3(x1_2_3)

        Yl_2, Yh_2 = self.xfm2(Yl_1)
        # x2_1 = Yh_2[0][:, :, 0, :, :].view(x.size(0), -1, 56, 56)
        # x2_2 = Yh_2[0][:, :, 1, :, :].view(x.size(0), -1, 56, 56)
        # x2_3 = Yh_2[0][:, :, 2, :, :].view(x.size(0), -1, 56, 56)
        x2_1 = F.interpolate(Yh_2[0][:, :, 0, :, :], size=(56, 56), mode='bilinear', align_corners=False)
        x2_2 = F.interpolate(Yh_2[0][:, :, 1, :, :], size=(56, 56), mode='bilinear', align_corners=False)
        x2_3 = F.interpolate(Yh_2[0][:, :, 2, :, :], size=(56, 56), mode='bilinear', align_corners=False)

        x2 = torch.cat((x2_1, x2_2, x2_3), dim=1)
        x2 = self.conv_x2(x2)

        x2_1 = x2[:, 0: 3, :, :]
        x2_2 = x2[:, 3: 6, :, :]
        x2_3 = x2[:, 6: 9, :, :]

        x2_1 = torch.cat((x2_1, x1_2_1), dim=1)
        x2_2 = torch.cat((x2_2, x1_2_2), dim=1)
        x2_3 = torch.cat((x2_3, x1_2_3), dim=1)

        x2_1 = self.conv_x2_1(x2_1)
        x2_2 = self.conv_x2_2(x2_2)
        x2_3 = self.conv_x2_3(x2_3)

        # three
        yl_5, yh_5 = self.xfm3(x2_1)
        yl_6, yh_6 = self.xfm3(x2_2)
        yl_7, yh_7 = self.xfm3(x2_3)

        x1_3_1 = torch.cat((yh_5[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28),
                            yh_6[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28),
                            yh_7[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28)), dim=1)
        x1_3_2 = torch.cat((yh_5[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28),
                            yh_6[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28),
                            yh_7[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28)), dim=1)
        x1_3_3 = torch.cat((yh_5[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28),
                            yh_6[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28),
                            yh_7[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28)), dim=1)

        x1_3_1 = self.conv_x1_3_1(x1_3_1)
        x1_3_2 = self.conv_x1_3_2(x1_3_2)
        x1_3_3 = self.conv_x1_3_3(x1_3_3)

        Yl_3, Yh_3 = self.xfm2(Yl_2)
        # x3_1 = Yh_3[0][:, :, 0, :, :].view(x.size(0), -1, 28, 28)
        # x3_2 = Yh_3[0][:, :, 1, :, :].view(x.size(0), -1, 28, 28)
        # x3_3 = Yh_3[0][:, :, 2, :, :].view(x.size(0), -1, 28, 28)
        x3_1 = F.interpolate(Yh_3[0][:, :, 0, :, :], size=(28, 28), mode='bilinear', align_corners=False)
        x3_2 = F.interpolate(Yh_3[0][:, :, 1, :, :], size=(28, 28), mode='bilinear', align_corners=False)
        x3_3 = F.interpolate(Yh_3[0][:, :, 2, :, :], size=(28, 28), mode='bilinear', align_corners=False)

        x3 = torch.cat((x3_1, x3_2, x3_3), dim=1)
        x3 = self.conv_x3(x3)

        x3_1 = x3[:, 0: 3, :, :]
        x3_2 = x3[:, 3: 6, :, :]
        x3_3 = x3[:, 6: 9, :, :]

        x3_1 = torch.cat((x3_1, x1_3_1), dim=1)
        x3_2 = torch.cat((x3_2, x1_3_2), dim=1)
        x3_3 = torch.cat((x3_3, x1_3_3), dim=1)

        x3_1 = self.conv_x3_1(x3_1)
        x3_2 = self.conv_x3_2(x3_2)
        x3_3 = self.conv_x3_3(x3_3)

        # four
        yl_8, yh_8 = self.xfm4(x3_1)
        yl_9, yh_9 = self.xfm4(x3_2)
        yl_10, yh_10 = self.xfm4(x3_3)

        x1_4_1 = torch.cat((yh_8[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14),
                            yh_9[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14),
                            yh_10[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14)), dim=1)
        x1_4_2 = torch.cat((yh_8[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14),
                            yh_9[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14),
                            yh_10[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14)), dim=1)
        x1_4_3 = torch.cat((yh_8[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14),
                            yh_9[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14),
                            yh_10[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14)), dim=1)

        x1_4_1 = self.conv_x1_4_1(x1_4_1)
        x1_4_2 = self.conv_x1_4_2(x1_4_2)
        x1_4_3 = self.conv_x1_4_3(x1_4_3)

        Yl_4, Yh_4 = self.xfm2(Yl_3)
        # x4_1 = Yh_4[0][:, :, 0, :, :].view(x.size(0), -1, 14, 14)
        # x4_2 = Yh_4[0][:, :, 1, :, :].view(x.size(0), -1, 14, 14)
        # x4_3 = Yh_4[0][:, :, 2, :, :].view(x.size(0), -1, 14, 14)
        x4_1 = F.interpolate(Yh_4[0][:, :, 0, :, :], size=(14, 14), mode='bilinear', align_corners=False)
        x4_2 = F.interpolate(Yh_4[0][:, :, 1, :, :], size=(14, 14), mode='bilinear', align_corners=False)
        x4_3 = F.interpolate(Yh_4[0][:, :, 2, :, :], size=(14, 14), mode='bilinear', align_corners=False)

        x4 = torch.cat((x4_1, x4_2, x4_3), dim=1)
        x4 = self.conv_x4(x4)

        x4_1 = x4[:, 0: 3, :, :]
        x4_2 = x4[:, 3: 6, :, :]
        x4_3 = x4[:, 6: 9, :, :]

        x4_1 = torch.cat((x4_1, x1_4_1), dim=1)
        x4_2 = torch.cat((x4_2, x1_4_2), dim=1)
        x4_3 = torch.cat((x4_3, x1_4_3), dim=1)

        x4_1 = self.conv_x4_1(x4_1)
        x4_2 = self.conv_x4_2(x4_2)
        x4_3 = self.conv_x4_3(x4_3)

        xh = torch.cat([x4_1, x4_2, x4_3], dim=1)
        xh = self.conv_xh(xh)

        features = self.xception.features(x)
        features = self.channel_attention(features)
        features = self.spatial_attention(features)
        x = self.channel_downsample(features)
        fusion2 = self.dual_cma1(x, xh)
        f2 = self.fusion1(fusion2[0], fusion2[1])
        f2 = F.relu(f2)
        f2 = F.adaptive_avg_pool2d(f2, (1, 1))
        f2 = f2.view(f2.size(0), -1)
        return f2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 提取特征（已包含ReLU、池化和展平）
        x = self.extract_features(x)

        # 应用二元分类器
        x = self.binary_classifier(x)

        # 应用Softmax获取每个类别的概率
        x = F.softmax(x, dim=1)

        return x


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    input = torch.rand([8, 3, 224, 224])
    net = HiFE(J=1, num_classes=2)
    output = net(input)
    print(output)
