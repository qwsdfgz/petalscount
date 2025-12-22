import torch.nn as nn
import torch
from torchvision import models




class SEAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(SEAttention, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False),
            nn.Sigmoid()  
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.fc(y)
        return x * y


class CSRNet(nn.Module):
    def __init__(self, load_weights=False):
        super(CSRNet, self).__init__()
        self.seen = 0
        self.frontend_feat = [
            64, 64, 'M', 
            128, 128, 'M', 
            256, 256, 256, 'M',  
            512, 512, 512, 'M',  
            512, 512, 512, 'M' 
        ]

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
        self.se = SEAttention(512) 
        self.output_layer = nn.Conv2d(64, 1, kernel_size=1)

        if not load_weights:
 
            mod = models.vgg16(pretrained=True)
            self._initialize_weights()
            for i, (name, param) in enumerate(self.frontend.state_dict().items()):
                vgg_param = list(mod.features.state_dict().items())[i]
                param.data[:] = vgg_param[1].data[:]

    def forward(self, x):
        x = self.frontend(x)  
        x = self.se(x)  
        x = self.backend(x)  
        x = self.output_layer(x)  
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