import torch
import torch.nn as nn
import torchvision.models as models

class DecoderBlock(nn.Module):
    """
    LinkNet 核心解码模块
    结构: 1x1 Conv (降维) -> Transpose Conv (上采样) -> 1x1 Conv (升维)
    """
    def __init__(self, in_channels, out_channels):
        super(DecoderBlock, self).__init__()
        
        # 内部降维比例，通常降为输入通道的 1/4
        inter_channels = in_channels // 4

        self.block = nn.Sequential(
            # 1. 1x1 卷积降维
            nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),

            # 2. 转置卷积上采样 (2倍)
            nn.ConvTranspose2d(inter_channels, inter_channels, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(inter_channels),
            nn.ReLU(inplace=True),

            # 3. 1x1 卷积恢复/调整通道数
            nn.Conv2d(inter_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

class LinkNet34(nn.Module):
    def __init__(self, output_ch=1):
        super(LinkNet34, self).__init__()

        # --- Encoder: ResNet34 ---
        # ResNet34 的 block expansion 是 1 (即 Layer 输出通道就是 64, 128, 256, 512)
        base_model = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)

        # 提取各个层 (Stem + Layers)
        self.initial_conv = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu,
            base_model.maxpool
        ) # Output: 64 ch, H/4 (ResNet的stem包含一个stride=2的conv和一个stride=2的maxpool)

        self.layer1 = base_model.layer1 # Output: 64 ch, H/4
        self.layer2 = base_model.layer2 # Output: 128 ch, H/8
        self.layer3 = base_model.layer3 # Output: 256 ch, H/16
        self.layer4 = base_model.layer4 # Output: 512 ch, H/32

        # --- Decoder ---
        # 对应 ResNet34 的通道: [64, 128, 256, 512]
        
        # Decoder 4: 输入 512 -> 输出 256 (为了和 Layer3 相加)
        self.decoder4 = DecoderBlock(512, 256)
        
        # Decoder 3: 输入 256 -> 输出 128 (为了和 Layer2 相加)
        self.decoder3 = DecoderBlock(256, 128)
        
        # Decoder 2: 输入 128 -> 输出 64 (为了和 Layer1 相加)
        self.decoder2 = DecoderBlock(128, 64)
        
        # Decoder 1: 输入 64 -> 输出 64
        # 这里不需要相加了，主要是完成 H/4 -> H/2 的上采样
        self.decoder1 = DecoderBlock(64, 64)

        # --- Final Head ---
        # 将 H/2 恢复到 H
        self.final_head = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_ch, kernel_size=1) # 最终分类
        )

    def forward(self, x):
        # --- Encoder Forward ---
        # Input: [B, 3, H, W]
        
        # Stem
        x0 = self.initial_conv(x) # [B, 64, H/4, W/4]
        
        # Layers
        e1 = self.layer1(x0)      # [B, 64, H/4, W/4]
        e2 = self.layer2(e1)      # [B, 128, H/8, W/8]
        e3 = self.layer3(e2)      # [B, 256, H/16, W/16]
        e4 = self.layer4(e3)      # [B, 512, H/32, W/32]

        # --- Decoder Forward (With Addition) ---
        
        # 1. e4 -> d4 + e3
        d4 = self.decoder4(e4)    # [B, 256, H/16, W/16]
        d4 = d4 + e3              # <--- LinkNet 特征: 直接相加 (Addition)
        
        # 2. d4 -> d3 + e2
        d3 = self.decoder3(d4)    # [B, 128, H/8, W/8]
        d3 = d3 + e2
        
        # 3. d3 -> d2 + e1
        d2 = self.decoder2(d3)    # [B, 64, H/4, W/4]
        d2 = d2 + e1
        
        # 4. d2 -> d1 (H/4 -> H/2)
        d1 = self.decoder1(d2)    # [B, 64, H/2, W/2]
        
        # 5. Final Upsampling (H/2 -> H)
        out = self.final_head(d1) # [B, output_ch, H, W]

        return out

# --- 测试代码 ---
if __name__ == '__main__':
    # 模拟输入
    dummy_input = torch.randn(2, 3, 512, 512)
    
    # 初始化模型
    model = LinkNet34(output_ch=3)
    
    # 前向计算
    output = model(dummy_input)
    
    print("输入尺寸:", dummy_input.shape)
    print("输出尺寸:", output.shape)
    
    assert output.shape == (2, 3, 512, 512), "输出尺寸不匹配！"
    print("LinkNet34 模型构建成功！")