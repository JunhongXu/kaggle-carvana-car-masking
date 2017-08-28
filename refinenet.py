import torch
import torch.nn as nn
import math
import numpy as np
from torchvision.models.resnet import Bottleneck


class RefineNetBlock(nn.Module):
    def __init__(self):
        super(RefineNetBlock, self).__init__()

    def forward(self, x):
        pass


class RCU(nn.Module):
    def __init__(self, input_feats, out_feats=256, batch_norm=False):
        """What is RCU output feature number? In paper, it says 256, but it can not sum with different input feature sizes."""
        super(RCU, self).__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(input_feats)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(input_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=not batch_norm)

        if batch_norm:
            self.bn2 = nn.BatchNorm2d(input_feats)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_feats, input_feats, kernel_size=3, stride=1, padding=1, bias=not batch_norm)

    def forward(self, x):
        if self.batch_norm:
            out = self.bn1(x)
        else:
            out = x
        out = self.relu1(out)
        out = self.conv1(out)

        if self.batch_norm:
            out = self.bn2(out)
        out = self.relu2(out)
        out = self.conv2(out)

        out += x
        return out


class FusionBlock(nn.Module):
    def __init__(self, input_feat1, input_feat2, size):
        """Fuse two input features, input_feat1 is the smaller feature"""
        super(FusionBlock, self).__init__()
        self.path1 = nn.Conv2d(input_feat1, input_feat1, kernel_size=3, stride=1, padding=1)
        self.path2 = nn.Conv2d(input_feat2, input_feat1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=size, mode='bilinear')

    def forward(self, path1, path2):
        x1 = self.path1(path1)
        x2 = self.path2(path2)

        return self.upsample(x1) + x2


class ChainedResPool(nn.Module):
    def __init__(self, num_feat):
        """how to do chained pooling exactly? every pool gives spatial size reduction, can not sum together different feature maps."""
        super(ChainedResPool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(x)
        out1 = self.pool1(x)
        out1 = self.conv1(out1)

        out = x + out1

        out2 = self.pool2(out1)
        out2 = self.conv2(out2)

        out = out + out2
        return out


class RefineNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        self.inplanes = 64
        super(RefineNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


if __name__ == '__main__':
    from torch.autograd import Variable
    a = Variable(torch.randn((1, 256, 32, 32)))
    rcu = ChainedResPool(256)
    print(rcu(a))