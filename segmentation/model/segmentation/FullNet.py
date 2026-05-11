"""
This script defines the structure of FullNet

Author: Hui Qu
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torchvision import models
from torchvision.models.vgg import VGG
from model.segmentation.DeepLabv3_plus import DeepLabv3_plus
class ConvLayer(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1,
                 groups=1):
        super(ConvLayer, self).__init__()
        self.add_module('conv', nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride,
                                          padding=padding, dilation=dilation, bias=False, groups=groups))
        self.add_module('relu', nn.LeakyReLU(inplace=True))
        self.add_module('bn', nn.BatchNorm2d(out_channels))


# --- different types of layers --- #
class BasicLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BasicLayer, self).__init__()
        self.conv = ConvLayer(in_channels, growth_rate, kernel_size=3, stride=1, padding=dilation,
                              dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv(x)
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


class BottleneckLayer(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, dilation=1):
        super(BottleneckLayer, self).__init__()

        inter_planes = growth_rate * 4
        self.conv1 = ConvLayer(in_channels, inter_planes, kernel_size=1, padding=0)
        self.conv2 = ConvLayer(inter_planes, growth_rate, kernel_size=3, padding=dilation, dilation=dilation)
        self.drop_rate = drop_rate

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        if self.drop_rate > 0:
            out = F.dropout(out, p=self.drop_rate, training=self.training)
        return torch.cat([x, out], 1)


# --- dense block structure --- #
class DenseBlock(nn.Sequential):
    def __init__(self, in_channels, growth_rate, drop_rate, layer_type, dilations):
        super(DenseBlock, self).__init__()
        for i in range(len(dilations)):
            layer = layer_type(in_channels+i*growth_rate, growth_rate, drop_rate, dilations[i])
            self.add_module('denselayer{:d}'.format(i+1), layer)


def choose_hybrid_dilations(n_layers, dilation_schedule, is_hybrid):
    import numpy as np
    # key: (dilation, n_layers)
    HD_dict = {(1, 4): [1, 1, 1, 1],
               (2, 4): [1, 2, 3, 2],
               (4, 4): [1, 2, 5, 9],
               (8, 4): [3, 7, 10, 13],
               (16, 4): [13, 15, 17, 19],
               (1, 6): [1, 1, 1, 1, 1, 1],
               (2, 6): [1, 2, 3, 1, 2, 3],
               (4, 6): [1, 2, 3, 5, 6, 7],
               (8, 6): [2, 5, 7, 9, 11, 14],
               (16, 6): [10, 13, 16, 17, 19, 21]}

    dilation_list = np.zeros((len(dilation_schedule), n_layers), dtype=np.int32)

    for i in range(len(dilation_schedule)):
        dilation = dilation_schedule[i]
        if is_hybrid:
            dilation_list[i] = HD_dict[(dilation, n_layers)]
        else:
            dilation_list[i] = [dilation for k in range(n_layers)]

    return dilation_list


class FullNet(nn.Module):
    def __init__(self, color_channels, output_channels=2, n_layers=6, growth_rate=24, compress_ratio=0.5,
                 drop_rate=0.1, dilations=(1,2,4,8,16,4,1), is_hybrid=True, layer_type='basic'):
        super(FullNet, self).__init__()
        if layer_type == 'basic':
            layer_type = BasicLayer
        else:
            layer_type = BottleneckLayer

        # 1st conv before any dense block
        in_channels = 24
        self.conv1 = ConvLayer(color_channels, in_channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers, dilations, is_hybrid)

        for i in range(n_blocks):  # no trans in last block
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block%d' % (i+1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans%d' % (i+1), trans)
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.conv2(out)
        return out


class FCN_pooling(nn.Module):
    """same structure with FullNet, except that there are pooling operations after block 1, 2, 3, 4
    and upsampling after block 5, 6
    """
    def __init__(self, color_channels, output_channels=2, n_layers=6, growth_rate=24, compress_ratio=0.5,
                 drop_rate=0.1, dilations=(1,2,4,8,16,4,1), hybrid=1, layer_type='basic'):
        super(FCN_pooling, self).__init__()
        if layer_type == 'basic':
            layer_type = BasicLayer
        else:
            layer_type = BottleneckLayer

        # 1st conv before any dense block
        in_channels = 24
        self.conv1 = ConvLayer(color_channels, in_channels, kernel_size=3, padding=1)

        self.blocks = nn.Sequential()
        n_blocks = len(dilations)

        dilation_list = choose_hybrid_dilations(n_layers, dilations, hybrid)

        for i in range(7):
            block = DenseBlock(in_channels, growth_rate, drop_rate, layer_type, dilation_list[i])
            self.blocks.add_module('block{:d}'.format(i+1), block)
            num_trans_in = int(in_channels + n_layers * growth_rate)
            num_trans_out = int(math.floor(num_trans_in * compress_ratio))
            trans = ConvLayer(num_trans_in, num_trans_out, kernel_size=1, padding=0)
            self.blocks.add_module('trans{:d}'.format(i+1), trans)
            if i in range(0, 4):
                self.blocks.add_module('pool{:d}'.format(i+1), nn.MaxPool2d(kernel_size=2, stride=2))
            elif i in range(4, 6):
                self.blocks.add_module('upsample{:d}'.format(i + 1), nn.UpsamplingBilinear2d(scale_factor=4))
            in_channels = num_trans_out

        # final conv
        self.conv2 = nn.Conv2d(in_channels, output_channels, kernel_size=3, stride=1,
                               padding=1, bias=False)
        # initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.conv1(x)
        out = self.blocks(out)
        out = self.conv2(out)
        return out
class UnetDsv3(nn.Module):
    def __init__(self, in_size, out_size, scale_factor):
        super(UnetDsv3, self).__init__()
        self.dsv = nn.Sequential(nn.Conv2d(in_size, out_size, kernel_size=1, stride=1, padding=0),
                                 nn.Upsample(size=scale_factor, mode='bilinear'), )

    def forward(self, input):
        return self.dsv(input)
    
class Unet_50(nn.Module):
    """
    使用预训练 ResNet-50 作为编码器的 U-Net 模型。
    """
    def __init__(self, in_ch, out_ch):
        super(Unet_50, self).__init__()
        print("正在构建 ResNet50UNet 模型")

        # --- 1. 编码器 (ResNet-50) ---
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        
        # 提取 ResNet 的层作为 U-Net 的编码器块：
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.pool1 = resnet.maxpool  # H/4 x W/4

        # ResNet50 各层通道数：layer1(256), layer2(512), layer3(1024), layer4(2048)
        self.encoder2 = resnet.layer1  # 256 channels, H/4 x W/4 (c2)
        self.encoder3 = resnet.layer2  # 512 channels, H/8 x W/8 (c3)
        self.encoder4 = resnet.layer3  # 1024 channels, H/16 x W/16 (c4)
        self.encoder5 = resnet.layer4  # 2048 channels, H/32 x W/32 (c5)

        # --- 2. 解码器 (Decoder) ---
        # 适配 ResNet50 通道数调整解码器卷积层通道
        
        # Decoder Block 4: 2048 -> 1024
        self.up_conv4 = nn.ConvTranspose2d(2048, 1024, 2, stride=2)
        self.decoder4_conv = DoubleConv(2048, 1024)  # 1024+1024=2048 输入通道

        # Decoder Block 3: 1024 -> 512
        self.up_conv3 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.decoder3_conv = DoubleConv(1024, 512)   # 512+512=1024 输入通道
        self.dsv2 = UnetDsv3(512, out_size=out_ch, scale_factor=(512, 512))  # Deep Supervision 2

        # Decoder Block 2: 512 -> 256
        self.up_conv2 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder2_conv = DoubleConv(512, 256)    # 256+256=512 输入通道
        self.dsv1 = UnetDsv3(256, out_size=out_ch, scale_factor=(512, 512))  # Deep Supervision 1

        # Decoder Block 1: 256 -> 64 (保持和原输入卷积层通道匹配)
        self.up_conv1 = nn.ConvTranspose2d(256, 64, 2, stride=2)
        self.decoder1_conv = DoubleConv(128, 64)     # 64+64=128 输入通道 (conv1输出是64通道)
        
        # --- 3. 输出层 (保持原逻辑) ---
        # 修复尺寸问题：添加最终的上采样层，将 H/2 x W/2 放大到 H x W
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 最后的 1x1 卷积
        self.finalconv = nn.Conv2d(64, out_ch, 1)

    def forward(self, x):
        # --------------------- 编码器 (Encoder) ---------------------
        c1 = self.conv1(x)      # 64 channels, H/2 x W/2
        p1 = self.pool1(c1)     # 64 channels, H/4 x W/4

        c2 = self.encoder2(p1)  # 256 channels, H/4 x W/4 (ResNet50 layer1 输出256通道)
        c3 = self.encoder3(c2)  # 512 channels, H/8 x W/8 (ResNet50 layer2 输出512通道)
        c4 = self.encoder4(c3)  # 1024 channels, H/16 x W/16 (ResNet50 layer3 输出1024通道)
        c5 = self.encoder5(c4)  # 2048 channels, H/32 x W/32 (ResNet50 layer4 输出2048通道)
        
        # --------------------- 解码器 (Decoder) ---------------------
        # Decoder Block 4
        up_4 = self.up_conv4(c5) 
        merge4 = torch.cat([up_4, c4], dim=1)  # 1024+1024=2048 通道
        d4 = self.decoder4_conv(merge4)        # 1024 channels, H/16 x W/16

        # Decoder Block 3
        up_3 = self.up_conv3(d4) 
        merge3 = torch.cat([up_3, c3], dim=1)  # 512+512=1024 通道
        d3 = self.decoder3_conv(merge3)        # 512 channels, H/8 x W/8
        out2 = self.dsv2(d3)                   # Deep Supervision output 2

        # Decoder Block 2
        up_2 = self.up_conv2(d3) 
        merge2 = torch.cat([up_2, c2], dim=1)  # 256+256=512 通道
        d2 = self.decoder2_conv(merge2)        # 256 channels, H/4 x W/4
        out1 = self.dsv1(d2)                   # Deep Supervision output 1

        # Decoder Block 1
        up_1 = self.up_conv1(d2) 
        merge1 = torch.cat([up_1, c1], dim=1)  # 64+64=128 通道
        d1 = self.decoder1_conv(merge1)        # 64 channels, H/2 x W/2
        
        # --------------------- 最终输出 (保持原逻辑) ---------------------
        # 1. 最终上采样到原始尺寸 (H x W)
        d_final = self.final_upsample(d1)      # 尺寸 H x W

        # 2. 最终 1x1 卷积
        c10 = self.finalconv(d_final)          # out_ch channels, 尺寸 H x W

        # 返回最终输出和深层监督输出
        return c10, out1, out2
class Unet_18(nn.Module):
    """
    使用预训练 ResNet-18 作为编码器的 U-Net 模型。
    """
    def __init__(self, in_ch, out_ch):
        super(Unet_18, self).__init__()
        print("正在构建 ResNet18UNet 模型")

        # --- 1. 编码器 (ResNet-18) ---
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 提取 ResNet 的层作为 U-Net 的编码器块：
        self.conv1 = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu
        )
        self.pool1 = resnet.maxpool # H/4 x W/4

        # ResNet 模块 (跳跃连接点)
        self.encoder2 = resnet.layer1  # 64 channels, H/4 x W/4 (c2)
        self.encoder3 = resnet.layer2  # 128 channels, H/8 x W/8 (c3)
        self.encoder4 = resnet.layer3  # 256 channels, H/16 x W/16 (c4)
        self.encoder5 = resnet.layer4  # 512 channels, H/32 x W/32 (c5)

        # --- 2. 解码器 (Decoder) ---
        
        # Decoder Block 4: 512 -> 256
        self.up_conv4 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.decoder4_conv = DoubleConv(512, 256) 

        # Decoder Block 3: 256 -> 128
        self.up_conv3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.decoder3_conv = DoubleConv(256, 128) 
        self.dsv2 = UnetDsv3(128, out_size=out_ch, scale_factor=(512, 512)) # Deep Supervision 2

        # Decoder Block 2: 128 -> 64
        self.up_conv2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.decoder2_conv = DoubleConv(128, 64) 
        self.dsv1 = UnetDsv3(64, out_size=out_ch, scale_factor=(512, 512)) # Deep Supervision 1

        # Decoder Block 1: 64 -> 64
        self.up_conv1 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.decoder1_conv = DoubleConv(128, 64) 
        
        # --- 3. 输出层 (已修改) ---
        
        # 修复尺寸问题：添加最终的上采样层，将 H/2 x W/2 放大到 H x W
        self.final_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 最后的 1x1 卷积
        self.finalconv = nn.Conv2d(64, out_ch, 1)


    def forward(self, x):
        # --------------------- 编码器 (Encoder) ---------------------
        c1 = self.conv1(x)      # 64 channels, H/2 x W/2
        p1 = self.pool1(c1)     # 64 channels, H/4 x W/4

        c2 = self.encoder2(p1)  # 64 channels, H/4 x W/4
        c3 = self.encoder3(c2)  # 128 channels, H/8 x W/8
        c4 = self.encoder4(c3)  # 256 channels, H/16 x W/16
        c5 = self.encoder5(c4)  # 512 channels, H/32 x W/32 (Bottleneck)
        
        # --------------------- 解码器 (Decoder) ---------------------

        # Decoder Block 4
        up_4 = self.up_conv4(c5) 
        merge4 = torch.cat([up_4, c4], dim=1) 
        d4 = self.decoder4_conv(merge4) # 256 channels, H/16 x W/16

        # Decoder Block 3
        up_3 = self.up_conv3(d4) 
        merge3 = torch.cat([up_3, c3], dim=1) 
        d3 = self.decoder3_conv(merge3) # 128 channels, H/8 x W/8
        out2 = self.dsv2(d3) # Deep Supervision output 2

        # Decoder Block 2
        up_2 = self.up_conv2(d3) 
        merge2 = torch.cat([up_2, c2], dim=1) 
        d2 = self.decoder2_conv(merge2) # 64 channels, H/4 x W/4
        out1 = self.dsv1(d2) # Deep Supervision output 1

        # Decoder Block 1
        up_1 = self.up_conv1(d2) 
        merge1 = torch.cat([up_1, c1], dim=1) 
        d1 = self.decoder1_conv(merge1) # 64 channels, H/2 x W/2
        
        # --------------------- 最终输出 (已修正) ---------------------
        
        # 1. 最终上采样到原始尺寸 (H x W)
        d_final = self.final_upsample(d1) # 尺寸 H x W

        # 2. 最终 1x1 卷积
        c10 = self.finalconv(d_final) # out_ch channels, 尺寸 H x W

        # 返回最终输出和深层监督输出
        return c10, out1, out2

class Unet01(nn.Module):
    def __init__(self,in_ch,out_ch):
        super(Unet01, self).__init__()
        print("Consturct Unet model")	
        self.conv1 = DoubleConv(in_ch, 64)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = DoubleConv(64, 128)
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = DoubleConv(128, 256)
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = DoubleConv(256, 512)
        self.pool4 = nn.MaxPool2d(2)
        self.conv5 = DoubleConv(512, 1024)
        self.up6 = nn.ConvTranspose2d(1024, 512, 2, stride=2)
        self.conv6 = DoubleConv(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.conv7 = DoubleConv(512, 256)
        self.dsv2 = UnetDsv3(256, out_size=3, scale_factor=(512, 512))
        self.up8 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv8 = DoubleConv(256, 128)
        self.dsv1 = UnetDsv3(128, out_size=3, scale_factor=(512, 512))
        self.up9 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv9 = DoubleConv(128, 64)
        self.conv10 = nn.Conv2d(64,out_ch, 1)
        self.finalconv = nn.Conv2d(64, 1, 3, padding=1)
        
    def forward(self,x):
        c1=self.conv1(x)
        p1=self.pool1(c1)
        c2=self.conv2(p1)
        p2=self.pool2(c2)
        c3=self.conv3(p2)
        p3=self.pool3(c3)
        c4=self.conv4(p3)
        p4=self.pool4(c4)
        c5=self.conv5(p4)
        up_6= self.up6(c5)
        merge6 = torch.cat([up_6, c4], dim=1)
        c6=self.conv6(merge6)
        up_7=self.up7(c4)
        merge7 = torch.cat([up_7, c3], dim=1)
        c7=self.conv7(merge7)
        out2=self.dsv2(c7)
        up_8=self.up8(c7)
        merge8 = torch.cat([up_8, c2], dim=1)
        c8=self.conv8(merge8)
        out1=self.dsv1(c8)
        up_9=self.up9(c8)
        merge9=torch.cat([up_9,c1],dim=1)
        c9=self.conv9(merge9)
        c10=self.conv10(c9)
    
        
      
        return c10,out1,out2


class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, input):
        return self.conv(input)

class Attention_block(nn.Module):
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block, self).__init__()
        self.w_g = nn.Sequential(
            nn.Conv2d(F_g,F_int,1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.w_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        # 下采样的gating signal 卷积
        g1 = self.w_g(g)
        # 上采样的 l 卷积
        x1 = self.w_x(x)
        # concat + relu
        psi = self.relu(g1 + x1)
        # channel 减为1，并Sigmoid,得到权重矩阵
        psi = self.psi(psi)
        # 返回加权的 x
        return x * psi




class FCN8s(nn.Module):

    def __init__(self, pretrained_net, n_class):
        print("Constructing FCN8s model...")
        super().__init__()
        self.n_class = n_class
        self.pretrained_net = pretrained_net
        self.relu = nn.ReLU(inplace=True)
        self.deconv1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn1 = nn.BatchNorm2d(512)
        self.deconv2 = nn.ConvTranspose2d(512, 256, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn2 = nn.BatchNorm2d(256)
        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.deconv4 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn4 = nn.BatchNorm2d(64)
        self.deconv5 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, dilation=1, output_padding=1)
        self.bn5 = nn.BatchNorm2d(32)
        self.classifier = nn.Conv2d(32, n_class, kernel_size=1)

    def forward(self, x):
        output = self.pretrained_net(x)
        x5 = output['x5']
        x4 = output['x4']
        x3 = output['x3']

        score = self.relu(self.deconv1(x5))
        score = self.bn1(score + x4)
        score = self.relu(self.deconv2(score))
        score = self.bn2(score + x3)
        score = self.bn3(self.relu(self.deconv3(score)))
        score = self.bn4(self.relu(self.deconv4(score)))
        score = self.bn5(self.relu(self.deconv5(score)))
        score = self.classifier(score)

        return score


class VGGNet(VGG):
    def __init__(self, pretrained=True, model='vgg16', requires_grad=True, remove_fc=True, show_params=False):
        super().__init__(make_layers(cfg[model]))
        self.ranges = ranges[model]

        if pretrained:
            exec("self.load_state_dict(models.%s(pretrained=True).state_dict())" % model)

        if not requires_grad:
            for param in super().parameters():
                param.requires_grad = False

        # delete redundant fully-connected layer params, can save memory
        # 去掉vgg最后的全连接层(classifier)
        if remove_fc:
            del self.classifier

        if show_params:
            for name, param in self.named_parameters():
                print(name, param.size())

    def forward(self, x):
        output = {}
        # get the output of each maxpooling layer (5 maxpool in VGG net)
        for idx, (begin, end) in enumerate(self.ranges):
            # self.ranges = ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)) (vgg16 examples)
            for layer in range(begin, end):
                x = self.features[layer](x)
            output["x%d" % (idx + 1)] = x

        return output


ranges = {
    'vgg11': ((0, 3), (3, 6), (6, 11), (11, 16), (16, 21)),
    'vgg13': ((0, 5), (5, 10), (10, 15), (15, 20), (20, 25)),
    'vgg16': ((0, 5), (5, 10), (10, 17), (17, 24), (24, 31)),
    'vgg19': ((0, 5), (5, 10), (10, 19), (19, 28), (28, 37))
}

# Vgg-Net config
# Vgg网络结构配置
cfg = {
    'vgg11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'vgg16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'vgg19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


# make layers using Vgg-Net config(cfg)
# 由cfg构建vgg-Net
def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

# if __name__ == '__main__':
#     net = Unet(3,3)
#     print(net)

if __name__ == "__main__":
    # 测试模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建经典UNet模型
    model = Unet_18(3,3).to(device)
    
    # 测试前向传播
    x = torch.randn(2, 3, 512, 512).to(device)
    c10, out1, out2 = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Main output shape: {c10.shape}")
    print(f"Aux output1 shape: {out1.shape}")
    print(f"Aux output2 shape: {out2.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")






