import torch
import torch.nn as nn
import math
import numpy as np
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls, model_zoo


class RefineNetBlock(nn.Module):
    def __init__(self, input_feats, output_feats, feat_size):
        super(RefineNetBlock, self).__init__()
        self.rcu_units = []
        for input_feat, out_feat in zip(input_feats, output_feats):
            rcu_1 = RCU(input_feat, out_feat)
            rcu_2 = RCU(out_feat, out_feat)
            self.rcu_units.append((rcu_1, rcu_2))

        if len(input_feats) > 1:
            self.fusion_block = FusionBlock(output_feats[0], output_feats[1], feat_size)

        self.pooling = ChainedResPool(output_feats[0])

    def forward(self, *x):
        if len(x) == 1:
            for (rcu_1, rcu_2) in self.rcu_units:
                x = rcu_1(x[0])
                x = rcu_2(x)
        else:
            xs = []
            for path, _x in enumerate(x):
                rcu_1, rcu_2 = self.rcu_units[path]
                xs.append(rcu_2(rcu_1(_x)))
            x = self.fusion_block(*xs)

        return self.pooling(x)


class RCU(nn.Module):
    def __init__(self, input_feats, out_feats=256, batch_norm=False):
        super(RCU, self).__init__()
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm2d(input_feats)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(input_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=not batch_norm)

        if batch_norm:
            self.bn2 = nn.BatchNorm2d(input_feats)
        self.relu2 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=not batch_norm)

        # add an additional conv layer to map input to the disred features
        self.conv3 = nn.Conv2d(input_feats, out_feats, 3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_feats)

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

        x = self.conv3(x)
        x = self.bn3(x)
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

        return self.upsample(x1) + self.upsample(x2)


class ChainedResPool(nn.Module):
    def __init__(self, num_feat):
        """how to do chained pooling exactly? every pool gives spatial size reduction,
        can not sum together different feature maps. make padding = 2"""
        super(ChainedResPool, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv1 = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)

        self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)

    def forward(self, x):
        x = self.relu(x)
        out1 = self.pool1(x)
        print(out1.size())
        out1 = self.conv1(out1)

        out = x + out1

        out2 = self.pool2(out1)
        out2 = self.conv2(out2)

        out = out + out2
        return out


class RefineNet1024(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(RefineNet1024, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])        # 256*256*256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 512*128*128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 1024*64*64
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2048*32*32

        self.refinenet4 = RefineNetBlock([2048], [512], 32*4)
        self.refinenet3 = RefineNetBlock([512, 1024], [256, 256], 64*4)
        self.refinenet2 = RefineNetBlock([256, 512], [256, 256], 128*4)
        self.refinenet1 = RefineNetBlock([256, 256], [256, 256], 256*4)

        self.rcu1 = RCU(input_feats=256, out_feats=32)
        self.rcu2 = RCU(input_feats=32, out_feats=1)
        # self.avgpool = nn.AvgPool2d(7)
        # self.fc = nn.Linear(512 * block.expansion, num_classes)

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

    def load_params(self):
        pretrained_dict = model_zoo.load_url(model_urls['resnet50'])
        model_dict = self.state_dict()
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        print(x1.size())
        x2 = self.layer2(x1)
        print(x2.size())
        x3 = self.layer3(x2)
        print(x3.size())
        x4 = self.layer4(x3)
        print(x4.size())

        x = self.refinenet4(x4)
        print(x.size())
        x = self.refinenet3(x, x3)
        print(x.size())
        x = self.refinenet2(x, x2)
        print(x.size())
        x = self.refinenet1(x, x1)
        print(x.size())

        x = self.rcu1(x)
        x = self.rcu2(x)
        # x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        # x = self.fc(x)

        return x


def test_refine_block(in_feat, out_feat, size):
    block = RefineNetBlock(in_feat, out_feat, size)
    print(block)
    x = Variable(torch.randn(1, 20, 16, 16))
    x1 = Variable(torch.randn(1, 32, 20, 20))
    print(block(x, x1))

if __name__ == '__main__':
    from torch.autograd import Variable
    # a = Variable(torch.randn((1, 256, 32, 32)))
    # rcu = ChainedResPool(256)
    # print(rcu(a))
    # test_refine_block([20, 32], [25, 32], 20)
    a = Variable(torch.randn((1, 3, 512, 512)))
    resnet = RefineNet1024(Bottleneck, [3, 4, 6, 3])
    resnet.load_params()
    # print(resnet(a))
    print(resnet)