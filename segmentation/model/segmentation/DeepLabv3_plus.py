'''
纯粹的deeplabv3+模型，无任何添加，支持resnet50和resnet01预训练模型。
'''
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ASPP_module(nn.Module):
    def __init__(self, inplanes, planes, os):
        super(ASPP_module, self).__init__()
        # ASPP
        if os == 16:
            dilations = [1, 6, 12, 18]
        elif os == 8:
            dilations = [1, 12, 24, 36]

        self.aspp1 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, padding=0, dilation=dilations[0], bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.aspp2 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.aspp3 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.aspp4 = nn.Sequential(
            nn.Conv2d(inplanes, planes, kernel_size=3, stride=1, padding=dilations[3], dilation=dilations[3], bias=False),
            nn.BatchNorm2d(planes),
            nn.ReLU()
        )
        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(2048, 256, 1, stride=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU()
        )

        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class DeepLabv3_plus(nn.Module):
    def __init__(self, nInputChannels=3, n_classes=3, os=16, resnet_type='101',_print=True):
        super(DeepLabv3_plus, self).__init__()
        if _print:
            print("Constructing DeepLabv3+ model...")
            print("Backbone: Resnet{}".format(resnet_type))
            print("Number of classes: {}".format(n_classes))
            print("Output stride: {}".format(os))
            print("Number of Input Channels: {}".format(nInputChannels))

        self.os = os

         # 选择 ResNet 主干
        if resnet_type == '101':
            backbone_fn = models.resnet101
            weight_path = './data/resnet101-cd907fc2.pth'
        elif resnet_type == '50':
            backbone_fn = models.resnet50
            weight_path = './data/resnet50-11ad3fa6.pth'
        else:
            raise ValueError("resnet_type must be '50' or '101'")
        
        # 尝试加载预训练模型
        try:
            self.resnet_features = backbone_fn(weights=None)
            pretrained_dict = torch.load(weight_path)
            self.resnet_features.load_state_dict(pretrained_dict)
            print(f"Loaded pre-trained ResNet-{resnet_type} loaded successfully.\n")
        except Exception as e:
            print(f"Failed to load pre-trained ResNet-{resnet_type}: {e}")
            self.resnet_features = backbone_fn(weights=models.ResNet50_Weights.DEFAULT if resnet_type == '50' else models.ResNet101_Weights.DEFAULT)
            print(f"Loaded ResNet-{resnet_type} with torchvision default weights instead.\n")

        # 修改 ResNet 的层以适应不同的输出步幅
        if os == 16:
            # 修改 layer3 和 layer4 的空洞率和填充
            self.resnet_features.layer3[0].conv2.dilation = (2, 2)
            self.resnet_features.layer3[0].conv2.padding = (2, 2)
            self.resnet_features.layer4[0].conv2.dilation = (4, 4)
            self.resnet_features.layer4[0].conv2.padding = (4, 4)
        elif os == 8:
            self.resnet_features.layer3[0].conv2.dilation = (4, 4)
            self.resnet_features.layer3[0].conv2.padding = (4, 4)
            self.resnet_features.layer4[0].conv2.dilation = (8, 8)
            self.resnet_features.layer4[0].conv2.padding = (8, 8)
    

        # ASPP
        self.ASPP = ASPP_module(2048, 256, os)

        # 解码器
        self.conv1 = nn.Conv2d(1280, 256, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

        # 低层特征处理
        self.conv2 = nn.Conv2d(256, 48, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(48)

        # 最后卷积层
        self.last_conv = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, n_classes, kernel_size=1, stride=1)
        )

    def forward(self, input):
        # 骨干网络前向传播
        x = self.resnet_features.conv1(input)
        x = self.resnet_features.bn1(x)
        x = self.resnet_features.relu(x)
        x = self.resnet_features.maxpool(x)
        x = self.resnet_features.layer1(x)
        low_level_features = x
        x = self.resnet_features.layer2(x)
        x = self.resnet_features.layer3(x)
        x = self.resnet_features.layer4(x)

        # ASPP 模块
        x = self.ASPP(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        # 上采样到低层次特征的尺寸
        x = F.interpolate(x, size=low_level_features.size()[2:], mode='bilinear', align_corners=True)

        # 融合低层次特征
        low_level_features = self.conv2(low_level_features)
        low_level_features = self.bn2(low_level_features)
        low_level_features = self.relu(low_level_features)
        x = torch.cat((x, low_level_features), dim=1)

        # 最后卷积层
        x = self.last_conv(x)

        # 上采样到输入尺寸
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

if __name__ == '__main__':
    # 初始化模型
    model = DeepLabv3_plus(nInputChannels=3, n_classes=3, os=8, resnet_type='50', _print=True)
    model.eval()  # 设置为评估模式

    # 创建一个随机输入张量（模拟一张 3 通道的 512x512 图像）
    input_tensor = torch.randn(1, 3, 512, 512)

    # 前向传播
    with torch.no_grad():
        output = model(input_tensor)

    # 打印输入和输出的形状
    # print(model)
    print("\n测试结果：")
    print(f"输入形状: {input_tensor.shape}")  # 应为 [1, 3, 512, 512]
    print(f"输出形状: {output.shape}")       # 应为 [1, n_classes, 512, 512]
