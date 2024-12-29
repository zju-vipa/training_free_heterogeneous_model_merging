import pdb
import torch
from torch import nn
import torchvision.transforms as T
import torchvision.models.vgg
# https://github.com/kuangliu/pytorch-cifar/blob/master/models/vgg.py
# cfg = {
#     'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
#     'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
#     'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
# }

cfg = {
    'VGG13': [128, 128, 128, 128, 'M2', 256, 256, 256, 256, 'M2', 512, 512, 512, 512, 'M'],
    'VGG19': [128, 128, 128, 128, 128, 128, 'M2', 256, 256, 256, 256, 256, 256, 'M2', 512, 512, 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    def __init__(self, vgg_name, w=1, num_classes=10):
        super(VGG, self).__init__()
        self.w = w
        self.features = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(self.w*512, num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        stage_blocks = []
        for x in cfg:
            if x == 'M':
                stage_blocks += [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers.append(nn.Sequential(*stage_blocks))
                stage_blocks = []
            elif x == 'M2':
                stage_blocks += [nn.MaxPool2d(kernel_size=4, stride=4)]
                layers.append(nn.Sequential(*stage_blocks))
                stage_blocks = []
            else:
                stage_blocks.append(nn.Sequential(
                    nn.Conv2d(in_channels if in_channels == 3 else self.w*in_channels,
                                     self.w*x, kernel_size=3, padding=1),
                    nn.BatchNorm2d(self.w*x),
                    nn.ReLU(inplace=True)
                ))
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

def vgg11(w=1, num_classes=10):
    return VGG('VGG11', w, num_classes=num_classes).cuda()

def vgg16(w=1, num_classes=10):
    return VGG('VGG16', w, num_classes=num_classes).cuda()

def vgg13(w=1, num_classes=10):
    return VGG('VGG13', w, num_classes=num_classes).cuda()

def vgg19(w=1, num_classes=10):
    return VGG('VGG19', w, num_classes=num_classes).cuda()
