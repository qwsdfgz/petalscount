import torch.nn as nn
import torch
from torchvision import models



# SE注意力机制模块
class SEAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SEAttention, self).__init__()
        # 全局平均池化将空间信息压缩为通道描述符
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # excitation操作：通过卷积层学习通道间的依赖关系（替代全连接层，保持维度一致性）
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()  # 输出通道注意力权重
        )

    def forward(self, x):
        # 计算每个通道的全局平均
        y = self.avg_pool(x)
        # 学习通道注意力权重
        y = self.fc(y)
        # 将注意力权重应用到输入特征图
        return x * y


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        # VGG16前端特征提取器配置（修正卷积层数量）
        self.frontend_feat = [
            64, 64, 'M',  # 第一块：2层卷积 + 池化
            128, 128, 'M',  # 第二块：2层卷积 + 池化
            256, 256, 256, 'M',  # 第三块：3层卷积 + 池化（VGG16特征）
            512, 512, 512, 'M',  # 第四块：3层卷积 + 池化（VGG16特征）
            512, 512, 512, 'M'  # 第五块：3层卷积 + 池化（VGG16特征）
        ]
        # 后端网络配置，用于上采样和特征融合
        self.backend_feat = [
            512, 512, 512,
            256,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            256, 128,
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            128, 64, 64
        ]

        self.frontend = make_layers(self.frontend_feat)
        self.backend = make_layers(self.backend_feat, in_channels=512, dilation=True)
        self.se = SEAttention(512)  # SE注意力（输入通道与VGG16输出匹配）
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
            # 加载预训练的VGG16模型权重（替换原VGG19）
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            # 将VGG16的权重迁移到前端网络
            for i, (name, param) in enumerate(self.frontend.state_dict().items()):
                vgg_param = list(mod.features.state_dict().items())[i]
                param.data[:] = vgg_param[1].data[:]

    def forward(self, x):
        x = self.frontend(x)  # VGG16特征提取
        x = self.se(x)  # 应用SE注意力机制
        x = self.backend(x)  # 后端上采样与特征融合
        x = self.output_layer(x)  # 输出密度图
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


def make_layers(cfg, in_channels=3, batch_norm=False, dilation=False):
    d_rate = 2 if dilation else 1
    layers = []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # 处理上采样层
        elif isinstance(v, nn.Upsample):
            layers += [v]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=d_rate, dilation=d_rate)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)