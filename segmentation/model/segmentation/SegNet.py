import torch
import torch.nn as nn
import torchvision.models as models

class SegNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=3, pretrained=True):
        """
        Args:
            input_channels: 输入通道数 (RGB=3)
            output_channels: 输出类别数 (腺体分割二分类=3)
            pretrained: 是否加载 VGG16_BN 的 ImageNet 预训练权重
        """
        super(SegNet, self).__init__()
        
        # --- Encoder (VGG16 Structure) ---
        # 我们必须手动定义层，以便在 forward 中获取 MaxPool 的 indices
        
        # Block 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 3
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 4
        self.enc4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)

        # Block 5
        self.enc5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)

        # --- Decoder (Symmetric to VGG16) ---
        self.unpool = nn.MaxUnpool2d(2, 2)

        # Block 5 Decoder
        self.dec5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Block 4 Decoder
        self.dec4 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Block 3 Decoder
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Block 2 Decoder
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Block 1 Decoder
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_channels, kernel_size=3, padding=1),
        )

        # 加载预训练权重
        if pretrained:
            self._load_vgg16_weights()

    def forward(self, x):
        # --- Encoder ---
        # 记录每层尺寸，用于 unpool 时处理奇数尺寸问题
        dim1 = x.size()
        x = self.enc1(x)
        x, idx1 = self.pool1(x)

        dim2 = x.size()
        x = self.enc2(x)
        x, idx2 = self.pool2(x)

        dim3 = x.size()
        x = self.enc3(x)
        x, idx3 = self.pool3(x)

        dim4 = x.size()
        x = self.enc4(x)
        x, idx4 = self.pool4(x)

        dim5 = x.size()
        x = self.enc5(x)
        x, idx5 = self.pool5(x)

        # --- Decoder ---
        x = self.unpool(x, idx5, output_size=dim5)
        x = self.dec5(x)

        x = self.unpool(x, idx4, output_size=dim4)
        x = self.dec4(x)

        x = self.unpool(x, idx3, output_size=dim3)
        x = self.dec3(x)

        x = self.unpool(x, idx2, output_size=dim2)
        x = self.dec2(x)

        x = self.unpool(x, idx1, output_size=dim1)
        x = self.dec1(x)

        return x

    def _load_vgg16_weights(self):
        """
        下载并加载 torchvision 官方的 VGG16_BN 权重到我们的 Encoder 中
        """
        print("正在加载 VGG16_BN 预训练权重...")
        try:
            # 获取官方预训练模型 (使用 weights 参数是新版 pytorch 标准，旧版使用 pretrained=True)
            vgg16_bn = models.vgg16_bn(weights=models.VGG16_BN_Weights.DEFAULT)
            pretrained_features = vgg16_bn.features
        except:
            # 兼容旧版本 PyTorch
            vgg16_bn = models.vgg16_bn(pretrained=True)
            pretrained_features = vgg16_bn.features

        # 定义我们的 encoder blocks 列表，顺序必须与 vgg16.features 中的层顺序对应
        # VGG16_BN features 结构是扁平的，我们需要按顺序对应到我们的 Block 中
        
        # 我们的 encoder 结构
        my_encoder_blocks = [self.enc1, self.enc2, self.enc3, self.enc4, self.enc5]
        
        # 指向 vgg16_bn.features 中的层索引
        vgg_layer_idx = 0
        
        # 遍历我们的 5 个 Encoder Block
        for block in my_encoder_blocks:
            # 遍历 Block 中的每一层 (Conv, BN, ReLU)
            for layer in block:
                # 只复制 Conv2d 和 BatchNorm2d 的参数，ReLU 不需要参数
                if isinstance(layer, (nn.Conv2d, nn.BatchNorm2d)):
                    # 确保 VGG 对应层也是相同类型
                    while not isinstance(pretrained_features[vgg_layer_idx], type(layer)):
                        vgg_layer_idx += 1
                        # 防止越界（理论上结构一致不会越界）
                        if vgg_layer_idx >= len(pretrained_features):
                            break
                    
                    if vgg_layer_idx < len(pretrained_features):
                        # 复制权重和偏置
                        src_layer = pretrained_features[vgg_layer_idx]
                        
                        layer.weight.data.copy_(src_layer.weight.data)
                        if layer.bias is not None and src_layer.bias is not None:
                            layer.bias.data.copy_(src_layer.bias.data)
                        
                        if isinstance(layer, nn.BatchNorm2d):
                            layer.running_mean.copy_(src_layer.running_mean)
                            layer.running_var.copy_(src_layer.running_var)
                        
                        vgg_layer_idx += 1
        
        print("VGG16_BN 预训练权重加载完成。")

if __name__ == "__main__":
    # 测试代码
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 实例化模型 (output_channels=1 用于腺体分割)
    model = SegNet(output_channels=3, pretrained=True).to(device)
    
    # 打印任意一层的权重均值，验证是否非零（说明加载成功）
    print(f"Layer 1 weight mean: {model.enc1[0].weight.mean().item()}")
    
    # 测试前向传播
    x = torch.randn(3, 3, 512, 512).to(device)
    y = model(x)
    print(f"Input: {x.shape}, Output: {y.shape}")