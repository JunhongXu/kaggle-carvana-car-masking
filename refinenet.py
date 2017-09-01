import torch
import torch.nn as nn
from torch.nn import functional as F
import math
import numpy as np
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models.resnet import model_urls, model_zoo


def make_conv_bn_relu(input_feat, output_feat, kernel_size=3, padding=1):

    layer = (nn.Conv2d(input_feat, output_feat, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
             nn.BatchNorm2d(output_feat), nn.ReLU())

    return layer


# class RefineNetBlock(nn.Module):
#     def __init__(self, input_feats, output_feats, feat_size):
#         super(RefineNetBlock, self).__init__()
#         if len(input_feats) == 2:
#             self.rcu1_1 = RCU(input_feats[0], output_feats[0])
#             self.rcu1_2 = RCU(output_feats[0], output_feats[0])
#
#             self.rcu2_1 = RCU(input_feats[1], output_feats[1])
#             self.rcu2_2 = RCU(output_feats[1], output_feats[1])
#         else:
#             self.rcu1_1 = RCU(input_feats[0], output_feats[0])
#             self.rcu1_2 = RCU(output_feats[0], output_feats[0])
#
#         if len(input_feats) > 1:
#             self.fusion_block = FusionBlock(output_feats[0], output_feats[1], feat_size)
#
#         self.pooling = ChainedResPool(output_feats[0])
#
#     def forward(self, *x):
#         if len(x) == 1:
#             x = self.rcu1_1(x[0])
#             x = self.rcu1_2(x)
#         else:
#             x1, x2 = x
#             rcu1, rcu2 = self.rcu1_2(self.rcu1_1(x1)), self.rcu2_2(self.rcu2_1(x2))
#             x = self.fusion_block(rcu1, rcu2)
#
#         return self.pooling(x)


class RCU(nn.Module):
    def __init__(self, input_feats, out_feats):
        super(RCU, self).__init__()
        #
        # self.bn1 = nn.BatchNorm2d(input_feats)
        # self.relu1 = nn.ReLU()
        # self.conv1 = nn.Conv2d(input_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=not batch_norm)
        self.layer1 = nn.Sequential(*make_conv_bn_relu(input_feats, out_feats))
        self.layer2 = nn.Sequential(*make_conv_bn_relu(out_feats, out_feats))
        self.transition = nn.Sequential(
            nn.Conv2d(input_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feats)
        )
        # if batch_norm:
        #     self.bn2 = nn.BatchNorm2d(input_feats)
        # self.relu2 = nn.ReLU()
        # self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=3, stride=1, padding=1, bias=not batch_norm)
        #
        # # add an additional conv layer to map input to the disred features
        # self.conv3 = nn.Conv2d(input_feats, out_feats, 3, stride=1, padding=1, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_feats)

    def forward(self, x):
        # if self.batch_norm:
        #     out = self.bn1(x)
        # else:
        #     out = x
        # out = self.relu1(out)
        # out = self.conv1(out)
        #
        # if self.batch_norm:
        #     out = self.bn2(out)
        # out = self.relu2(out)
        # out = self.conv2(out)
        #
        # x = self.conv3(x)
        # x = self.bn3(x)
        out = self.layer1(x)
        out = self.layer2(out)
        x = self.transition(x)
        out += x
        return F.relu(out)


# class FusionBlock(nn.Module):
#     def __init__(self, input_feat1, input_feat2, size):
#         """Fuse two input features, input_feat1 is the smaller feature"""
#         super(FusionBlock, self).__init__()
#         self.path1 = nn.Conv2d(input_feat1, input_feat1, kernel_size=3, stride=1, padding=1)
#         self.path2 = nn.Conv2d(input_feat2, input_feat1, kernel_size=3, stride=1, padding=1)
#         self.upsample = nn.Upsample(size=size, mode='bilinear')
#
#     def forward(self, path1, path2):
#         x1 = self.path1(path1)
#         x2 = self.path2(path2)
#
#         return self.upsample(x1) + self.upsample(x2)


# class ChainedResPool(nn.Module):
#     def __init__(self, num_feat):
#         """how to do chained pooling exactly? every pool gives spatial size reduction,
#         can not sum together different feature maps. make padding = 2"""
#         super(ChainedResPool, self).__init__()
#         self.relu = nn.ReLU(inplace=True)
#         self.pool1 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
#         self.conv1 = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)
#
#         self.pool2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
#         self.conv2 = nn.Conv2d(num_feat, num_feat, 3, stride=1, padding=1)
#
#     def forward(self, x):
#         x = self.relu(x)
#         out1 = self.pool1(x)
#         out1 = self.conv1(out1)
#
#         out = x + out1
#
#         out2 = self.pool2(out1)
#         out2 = self.conv2(out2)
#
#         out = out + out2
#         return out


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

        self.middle = nn.Sequential(*make_conv_bn_relu(2048, 1024, 1, padding=0))     # 1024*32*32
        self.maxpool2 = nn.MaxPool2d(2, 2)                                # 1024*16*16

        self.refinenet3 = RCU(2048, 512)       # 512*64*64
        self.trans3 = nn.Sequential(*make_conv_bn_relu(1024, 1024, 1, padding=0))

        self.refinenet2 = RCU(1024, 256)        # 256*128*128
        self.trans2 = nn.Sequential(*make_conv_bn_relu(512, 512, 1, padding=0))

        self.refinenet1 = RCU(512, 64)         # 64*256*256
        self.trans1 = nn.Sequential(*make_conv_bn_relu(256, 256, 1, padding=0))

        self.refinenet0 = RCU(64, 64)          # 64*512*512
        self.trans0 = nn.Sequential(*make_conv_bn_relu(64, 64, 1, padding=0))

        self.final = nn.Sequential(
            *make_conv_bn_relu(64, 32),
            *make_conv_bn_relu(32, 16),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(16, 1, kernel_size=3, stride=1, padding=1)
        )

    def set_requires_grad(self, layer):
        for param in layer.parameters():
            param.requires_grad = False

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

    def load_params(self, resnet='resnet50'):
        pretrained_dict = model_zoo.load_url(model_urls[resnet])
        model_dict = self.state_dict()
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 64*256*256

        x1 = self.layer1(x)     # 128*256*256
        x2 = self.layer2(x1)    # 512*128*128
        x3 = self.layer3(x2)    # 1024*64*64
        x4 = self.layer4(x3)    # 2048*32*32

        x4 = self.middle(x4)    # 1024*32*32

        out = F.upsample(x4, scale_factor=2, mode='bilinear')   # 1024*64*64
        x3 = self.trans3(x3)
        out = torch.cat((x3, out), 1)   # 2048*64*64
        out = self.refinenet3(out)      # 512*64*64

        out = F.upsample(out, scale_factor=2, mode='bilinear')  # 512*128*128
        x2 = self.trans2(x2)
        out = torch.cat((x2, out), 1)                           # 1024*128*128
        out = self.refinenet2(out)                              # 256*128*128

        out = F.upsample(out, scale_factor=2, mode='bilinear')  # 256*256*256
        x1 = self.trans1(x1)
        out = torch.cat((x1, out), 1)                           # 512*256*256
        out = self.refinenet1(out)                              # 64*256*256

        # out = F.upsample(out, scale_factor=2, mode='bilinear')  # 64*512*512
        x = self.trans0(x)
        out = x + out
        out = self.refinenet0(out)                              # 64*256*256
        out = F.upsample(out, mode='bilinear', scale_factor=2)

        out = self.final(out)

        return out, F.sigmoid(out)


def test_refine_block(in_feat, out_feat, size):
    block = RefineNetBlock(in_feat, out_feat, size)
    print(block)
    x = Variable(torch.randn(1, 20, 16, 16))
    x1 = Variable(torch.randn(1, 32, 20, 20))
    print(block(x, x1))

if __name__ == '__main__':
    from torch.autograd import Variable
    a = Variable(torch.randn((4, 3, 1024, 1024))).cuda()
    resnet = RefineNet1024(Bottleneck, [3, 4, 6, 3]).cuda()
    # resnet.load_params()
    resnet = nn.DataParallel(resnet)
    # resnet.cuda()
    # print(resnet(a))
    print(resnet(a))
    print(resnet)
