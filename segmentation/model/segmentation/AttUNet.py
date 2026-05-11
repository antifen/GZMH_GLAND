import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvolutionBlock(nn.Module):
    """
    解码器中使用的标准卷积块
    (Conv -> BN -> ReLU) * 2
    """
    def __init__(self, in_ch, out_ch):
        super(ConvolutionBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class AttentionBlock(nn.Module):
    """
    Attention Gate
    用于过滤 Skip Connection 的特征
    """
    def __init__(self, F_g, F_l, F_int):
        super(AttentionBlock, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )
        
        self.W_x = nn.Sequential(
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
        # g: Gating Signal (来自解码器下层，已经上采样过)
        # x: Skip Connection (来自编码器)
        
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class UpBlock(nn.Module):
    """
    上采样模块：Upsample -> Attention -> Concat -> ConvBlock
    """
    def __init__(self, in_ch, skip_ch, out_ch):
        super(UpBlock, self).__init__()
        
        # 使用双线性插值上采样，也可以改用 ConvTranspose2d
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        
        # 注意力门
        # F_g (gating) = in_ch (从下层up上来的)
        # F_l (local) = skip_ch (从左边skip过来的)
        self.att = AttentionBlock(F_g=in_ch, F_l=skip_ch, F_int=in_ch // 2)
        
        # 拼接后的卷积: 输入通道 = 上采样后的通道 + Skip通道
        self.conv = ConvolutionBlock(in_ch + skip_ch, out_ch)

    def forward(self, x, skip_x):
        x = self.up(x)
        
        # 如果输入尺寸因为奇数填充导致不匹配，进行padding (增强鲁棒性)
        if x.shape != skip_x.shape:
            x = F.interpolate(x, size=skip_x.shape[2:], mode='bilinear', align_corners=True)
            
        # 注意力机制处理 Skip Connection
        skip_x_att = self.att(g=x, x=skip_x)
        
        # 拼接
        x = torch.cat([skip_x_att, x], dim=1)
        
        return self.conv(x)

class AttUNet_ResNet18(nn.Module):
    def __init__(self, output_ch=1):
        super(AttUNet_ResNet18, self).__init__()

        # --- Encoder: ResNet18 ---
        # 加载预训练模型
        base_model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        
        # 拆解 ResNet 的层用于 Skip Connections
        self.initial_conv = nn.Sequential(
            base_model.conv1,
            base_model.bn1,
            base_model.relu
        ) # 输出: 64 ch, H/2
        
        self.maxpool = base_model.maxpool # 输出: 64 ch, H/4
        self.layer1 = base_model.layer1   # 输出: 64 ch, H/4
        self.layer2 = base_model.layer2   # 输出: 128 ch, H/8
        self.layer3 = base_model.layer3   # 输出: 256 ch, H/16
        self.layer4 = base_model.layer4   # 输出: 512 ch, H/32 (Bottleneck)

        # --- Decoder ---
        # ResNet18 的通道数分别为: [64, 64, 128, 256, 512]
        
        # Decoder 4: Input 512 -> Up -> Join Skip3 (256) -> Out 256
        self.up4 = UpBlock(in_ch=512, skip_ch=256, out_ch=256)
        
        # Decoder 3: Input 256 -> Up -> Join Skip2 (128) -> Out 128
        self.up3 = UpBlock(in_ch=256, skip_ch=128, out_ch=128)
        
        # Decoder 2: Input 128 -> Up -> Join Skip1 (64) -> Out 64
        # 注意：ResNet layer1 输出是 64通道
        self.up2 = UpBlock(in_ch=128, skip_ch=64, out_ch=64)
        
        # Decoder 1: Input 64 -> Up -> Join Initial_Conv (64) -> Out 64
        self.up1 = UpBlock(in_ch=64, skip_ch=64, out_ch=64)
        
        # Final Upsampling to restore original resolution (H/2 -> H)
        self.final_up = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, output_ch, kernel_size=1)
        )

    def forward(self, x):
        # --- Encoder Forward ---
        # x: [B, 3, H, W]
        
        x0 = self.initial_conv(x) # Skip 1: [B, 64, H/2, W/2]
        x1 = self.maxpool(x0)     
        x1 = self.layer1(x1)      # Skip 2: [B, 64, H/4, W/4]
        x2 = self.layer2(x1)      # Skip 3: [B, 128, H/8, W/8]
        x3 = self.layer3(x2)      # Skip 4: [B, 256, H/16, W/16]
        x4 = self.layer4(x3)      # Bridge: [B, 512, H/32, W/32]
        
        # --- Decoder Forward ---
        d4 = self.up4(x4, x3)     # -> [B, 256, H/16, W/16]
        d3 = self.up3(d4, x2)     # -> [B, 128, H/8, W/8]
        d2 = self.up2(d3, x1)     # -> [B, 64, H/4, W/4]
        d1 = self.up1(d2, x0)     # -> [B, 64, H/2, W/2]
        
        out = self.final_up(d1)   # -> [B, out_ch, H, W]
        
        return out

# --- 简单的测试代码 ---
if __name__ == '__main__':
    # 模拟输入：Batch=2, Channel=3, 256x256
    # ResNet默认输入是3通道
    dummy_input = torch.randn(2, 3, 512, 512)
    
    # 初始化模型，使用预训练权重
    model = AttUNet_ResNet18(output_ch=3)
    
    # 打印参数量信息
    # print(model) 
    
    output = model(dummy_input)
    
    print("输入尺寸:", dummy_input.shape)
    print("输出尺寸:", output.shape)
    
    assert output.shape == (2, 3, 512, 512), "输出尺寸不对！"
    print("AttUNet (ResNet18 Backbone) 构建成功！")