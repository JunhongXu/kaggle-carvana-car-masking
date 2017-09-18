import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import Bottleneck, BasicBlock
from torchvision.models.resnet import model_urls as resnet_urls
from torchvision.models.resnet import model_zoo as resnet_zoo
from torchvision.models.vgg import cfg, make_layers
from torch.autograd import Variable
from torchvision.models.vgg import model_urls as vgg_urls
from torchvision.models.vgg import model_zoo as vgg_zoo


def make_conv_bn_relu(input_feat, output_feat, kernel_size=3, padding=1, inplace=False, use_bias=False):
    layer = (nn.Conv2d(input_feat, output_feat, kernel_size=kernel_size, stride=1, padding=padding, bias=use_bias),
             nn.BatchNorm2d(output_feat), nn.ReLU(inplace=inplace))

    return layer


class RCU(nn.Module):
    def __init__(self, input_feats, out_feats):
        super(RCU, self).__init__()
        self.layer1 = nn.Sequential(*make_conv_bn_relu(input_feats, out_feats))
        self.layer2 = nn.Sequential(*make_conv_bn_relu(out_feats, out_feats))
        self.transition = nn.Sequential(
            nn.Conv2d(input_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feats)
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        x = self.transition(x)
        out += x
        return F.relu(out)


class GateUnit(nn.Module):
    def __init__(self, in_feat, out_feat):
        """First is a smaller feature and second is a larger feature"""
        super(GateUnit, self).__init__()
        self.conv1 = nn.Conv2d(in_feat, in_feat, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_feat)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(out_feat, in_feat, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_feat)
        self.relu2 = nn.ReLU(inplace=True)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, x1, x2):
        """x1 is smaller feature map"""
        x1 = self.relu1(self.bn1(self.conv1(x1)))
        x1 = self.upsample(x1)

        x2 = self.relu2(self.bn2(self.conv2(x2)))
        return torch.mul(x1, x2)


class GatedRefinementUnit(nn.Module):
    def __init__(self, in_feat):
        super(GatedRefinementUnit, self).__init__()
        self.conv1 = nn.Sequential(*make_conv_bn_relu(in_feat, 1))
        self.conv2 = nn.Sequential(*make_conv_bn_relu(2, 1))
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

    def forward(self, map, x):
        conv1 = self.conv1(x)
        map = torch.cat((conv1, map), 1)
        map = self.conv2(map)
        map = self.upsample(map)
        return map


class RefineNetV6(nn.Module):
    """VGG16-BN with gated info between two encoder layers with auxiliary loss """
    def __init__(self):
        super(RefineNetV6, self).__init__()
        self.layer1_1 = nn.Sequential(*make_conv_bn_relu(3, 64, inplace=True, use_bias=True))
        self.layer1_2 = nn.Sequential(*make_conv_bn_relu(64, 64, inplace=True, use_bias=True))

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 64*512

        self.layer2_1 = nn.Sequential(*make_conv_bn_relu(64, 128, inplace=True, use_bias=True))
        self.layer2_2 = nn.Sequential(*make_conv_bn_relu(128, 128, inplace=True, use_bias=True))

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 128*256

        self.layer3_1 = nn.Sequential(*make_conv_bn_relu(128, 256, inplace=True, use_bias=True))
        self.layer3_2 = nn.Sequential(*make_conv_bn_relu(256, 256, inplace=True, use_bias=True))
        self.layer3_3 = nn.Sequential(*make_conv_bn_relu(256, 256, inplace=True, use_bias=True))

        self.maxpool3 = nn.MaxPool2d(2, 2)                     # 256*128

        self.layer4_1 = nn.Sequential(*make_conv_bn_relu(256, 512, inplace=True, use_bias=True))
        self.layer4_2 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))
        self.layer4_3 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))

        self.maxpool4 = nn.MaxPool2d(2, 2)                     # 512*64

        self.layer5_1 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))
        self.layer5_2 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))
        self.layer5_3 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))

        self.maxpool5 = nn.MaxPool2d(2, 2)                     # 512*32

        self.layer6 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=False), nn.MaxPool2d(2, 2)) # 512*16

        self.middle = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=False)) # 512*16

        self.map1 = nn.Sequential(*make_conv_bn_relu(512, 1, inplace=True, use_bias=False), nn.Upsample(scale_factor=2, mode='bilinear'))
        self.gate1 = GateUnit(512, 512)
        self.map2 = GatedRefinementUnit(512)

        self.gate2 = GateUnit(512, 512)
        self.map3 = GatedRefinementUnit(512)

        self.gate3 = GateUnit(512, 256)
        self.map4 = GatedRefinementUnit(256)

        self.gate4 = GateUnit(256, 128)
        self.map5 = GatedRefinementUnit(128)

        self.gate5 = GateUnit(128, 64)
        self.map6 = GatedRefinementUnit(64)

        self.map7 = nn.Sequential(
            *make_conv_bn_relu(1, 64),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(63, 1, 3, stride=1, padding=1)
        )

    def forward(self, x):
        x1 = self.maxpool1(self.layer1_2(self.layer1_1(x)))
        x2 = self.maxpool2(self.layer2_2(self.layer2_1(x1)))
        x3 = self.maxpool3(self.layer3_3(self.layer3_2(self.layer3_1(x2))))
        x4 = self.maxpool4(self.layer4_3(self.layer4_2(self.layer4_1(x3))))
        x5 = self.maxpool5(self.layer5_3(self.layer5_2(self.layer5_1(x4))))
        x6 = self.layer6(x5)

        middle = self.middle(x6)

        map1 = self.map1(middle)
        gate1 = self.gate1(x6, x5)
        map2 =self.map2(map1, gate1)

        gate2 = self.gate2(gate1, x4)
        map3 = self.map3(map2, gate2)

        gate3 = self.gate3(gate2, x3)
        map4 = self.map4(map3, gate3)

        gate4 = self.gate4(gate3, x2)
        map5 = self.map5(map4, gate4)

        gate5 = self.gate5(gate4, x1)
        map6 = self.map6(map5, gate5)

        map7 = self.map7(map6)
        return (map7, F.sigmoid(map7)), (map1, map2, map3, map4, map5, map6, map7)


class RefineNetV5_1024(nn.Module):
    """VGG16-BN on each decoder stage"""
    def __init__(self):
        super(RefineNetV5_1024, self).__init__()
        self.layer1_1 = nn.Sequential(*make_conv_bn_relu(3, 64, inplace=True, use_bias=True))
        self.layer1_2 = nn.Sequential(*make_conv_bn_relu(64, 64, inplace=True, use_bias=True))

        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)   # 64*512

        self.layer2_1 = nn.Sequential(*make_conv_bn_relu(64, 128, inplace=True, use_bias=True))
        self.layer2_2 = nn.Sequential(*make_conv_bn_relu(128, 128, inplace=True, use_bias=True))

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)   # 128*256

        self.layer3_1 = nn.Sequential(*make_conv_bn_relu(128, 256, inplace=True, use_bias=True))
        self.layer3_2 = nn.Sequential(*make_conv_bn_relu(256, 256, inplace=True, use_bias=True))
        self.layer3_3 = nn.Sequential(*make_conv_bn_relu(256, 256, inplace=True, use_bias=True))

        self.maxpool3 = nn.MaxPool2d(2, 2)                     # 256*128

        self.layer4_1 = nn.Sequential(*make_conv_bn_relu(256, 512, inplace=True, use_bias=True))
        self.layer4_2 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))
        self.layer4_3 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))

        self.maxpool4 = nn.MaxPool2d(2, 2)                     # 512*64

        self.layer5_1 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))
        self.layer5_2 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))
        self.layer5_3 = nn.Sequential(*make_conv_bn_relu(512, 512, inplace=True, use_bias=True))

        self.maxpool5 = nn.MaxPool2d(2, 2)                     # 512*32

        self.middle = nn.Sequential(*make_conv_bn_relu(512, 1024, inplace=True, use_bias=False), nn.MaxPool2d(2, 2)) # 16

        self.up_1 = RCU(1024, 512)                             # 512
        # self.trans1 = RCU(512, 512)
        self.up_2 = RCU(1024, 256)                             # 256
        # self.trans2 = RCU(512, 512)
        self.up_3 = RCU(768, 128)                              # 128
        # self.trans3 = RCU(256, 256)
        self.up_4 = RCU(384, 64)                               # 64
        # self.trans4 = RCU(128, 128)
        self.up_5 = RCU(192, 32)   # 32
        # self.trans5 = RCU(64, 64)
        self.final = nn.Sequential(
            *make_conv_bn_relu(96, 16),
            nn.Upsample(mode='bilinear', scale_factor=2),
            nn.Conv2d(16, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x1 = self.maxpool1(self.layer1_2(self.layer1_1(x)))
        x2 = self.maxpool2(self.layer2_2(self.layer2_1(x1)))
        x3 = self.maxpool3(self.layer3_3(self.layer3_2(self.layer3_1(x2))))
        x4 = self.maxpool4(self.layer4_3(self.layer4_2(self.layer4_1(x3))))
        x5 = self.maxpool5(self.layer5_3(self.layer5_2(self.layer5_1(x4))))

        middle = self.middle(x5)

        out = self.up_1(middle) # 512
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        # x5 = self.trans1(x5)    # 512
        out = torch.cat((out, x5), 1)

        out = self.up_2(out)    # 256
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        # x4 = self.trans2(x4)    # 512
        out = torch.cat((out, x4), 1)

        out = self.up_3(out)    # 128
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        # x3 = self.trans3(x3)   # 256
        out = torch.cat((out, x3), 1)

        out = self.up_4(out)    # 64
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        # x2 = self.trans4(x2)    # 128
        out = torch.cat((out, x2), 1)

        out = self.up_5(out)    # 32
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        # x1 = self.trans5(x1)    # 64
        out = torch.cat((out, x1), 1)

        scores = self.final(out)
        return scores, F.sigmoid(scores)

    def load_vgg16(self):
        model_dict = self.state_dict()
        pretrain_dict = vgg_zoo.load_url(vgg_urls['vgg16_bn'])

        model_dict.update(
            {key: pretrain_dict[pretrain_key] for key, pretrain_key in zip(model_dict.keys(), pretrain_dict.keys()) if
             'classifier' not in pretrain_key})

        self.load_state_dict(model_dict)


class RefineNetV4_1024(nn.Module):
    """ResNet-38"""
    def __init__(self, block, layers):
        self.inplanes = 64
        super(RefineNetV4_1024, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # 64*256*256
        self.layer1 = self._make_layer(block, 64, layers[0])            # 64*256*256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 128*128*128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 256*64*64
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 512*32*32

        self.middle = nn.Sequential(*make_conv_bn_relu(512, 1024, 1, padding=0), nn.MaxPool2d(2, 2)) # 1024*32*32

        self.refinenet3 = nn.Sequential(RCU(1024+512, 1024), RCU(1024, 512))       # 512*64*64
        self.trans3 = nn.Sequential(*make_conv_bn_relu(512, 512, 1, padding=0))

        self.refinenet2 = nn.Sequential(RCU(512+256, 512), RCU(512, 256))        # 256*128*128
        self.trans2 = nn.Sequential(*make_conv_bn_relu(256, 256, 1, padding=0))

        self.refinenet1 = nn.Sequential(RCU(256+128, 256), RCU(256, 128))       # 64*256*256
        self.trans1 = nn.Sequential(*make_conv_bn_relu(128, 128, 1, padding=0))

        self.refinenet0 = nn.Sequential(RCU(128+64, 128), RCU(128, 64)) #, RCU(64, 64))        # 64*512*512
        self.trans0 = nn.Sequential(*make_conv_bn_relu(64, 64, 1, padding=0))

        self.final_0 = nn.Sequential(RCU(64+64, 64), RCU(64, 64))
        self.final_trans_0 = nn.Sequential(*make_conv_bn_relu(64, 64, 1, padding=0))

        self.final = nn.Sequential(
            *make_conv_bn_relu(64, 16),
            *make_conv_bn_relu(16, 8),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
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
        pretrained_dict = resnet_zoo.load_url(resnet_urls[resnet])
        model_dict = self.state_dict()
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)     # 64*256*256

        x1 = self.layer1(x)     # 64*256*256
        x2 = self.layer2(x1)    # 128*128*128
        x3 = self.layer3(x2)    # 256*64*64
        x4 = self.layer4(x3)    # 512*32*32

        # print(x.size())
        # print(x1.size())
        # print(x2.size())
        # print(x3.size())
        # print(x4.size())

        middle = self.middle(x4)    # 1024*32*32

        out = F.upsample(middle, scale_factor=2, mode='bilinear')   # 1024*64*64
        x4 = self.trans3(x4)
        out = torch.cat((x4, out), 1)   # 1024+512*64*64
        out = self.refinenet3(out)      # 512*64*64

        out = F.upsample(out, scale_factor=2, mode='bilinear')  # 512*128*128
        x3 = self.trans2(x3)
        out = torch.cat((x3, out), 1)                           # 1024*128*128
        out = self.refinenet2(out)                              # 256*128*128

        out = F.upsample(out, scale_factor=2, mode='bilinear')  # 256*256*256
        x2 = self.trans1(x2)
        out = torch.cat((x2, out), 1)                           # 512*256*256
        out = self.refinenet1(out)                              # 64*256*256

        # out = F.upsample(out, scale_factor=2, mode='bilinear')  # 64*512*512
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        x1 = self.trans0(x1)
        out = torch.cat((x1, out), 1)
        out = self.refinenet0(out)                              # 64*256*256

        x = self.final_trans_0(x)
        out = torch.cat((x, out), 1)
        out = self.final_0(out)
        out = F.upsample(out, scale_factor=2, mode='bilinear')

        out = self.final(out)

        return out, F.sigmoid(out)


class RefineNetV3_1024(nn.Module):
    """ResNet-50"""
    def __init__(self, block, layers):
        self.inplanes = 64
        super(RefineNetV3_1024, self).__init__()
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

        self.refinenet3 = nn.Sequential(RCU(2048, 512), RCU(512, 512))       # 512*64*64
        self.trans3 = nn.Sequential(*make_conv_bn_relu(1024, 1024, 1, padding=0))

        self.refinenet2 = nn.Sequential(RCU(1024, 256), RCU(256, 256))        # 256*128*128
        self.trans2 = nn.Sequential(*make_conv_bn_relu(512, 512, 1, padding=0))

        self.refinenet1 = nn.Sequential(RCU(512, 64), RCU(64, 64))       # 64*256*256
        self.trans1 = nn.Sequential(*make_conv_bn_relu(256, 256, 1, padding=0))

        self.refinenet0 = nn.Sequential(RCU(64, 64)) #, RCU(64, 64))        # 64*512*512
        self.trans0 = nn.Sequential(*make_conv_bn_relu(64, 64, 1, padding=0))

        self.final = nn.Sequential(
            *make_conv_bn_relu(64, 16),
            *make_conv_bn_relu(16, 8),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(8, 1, kernel_size=3, stride=1, padding=1)
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
        pretrained_dict = resnet_zoo.load_url(resnet_urls[resnet])
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


class RefineNetV2_1024(nn.Module):
    """ResNet-50"""
    def __init__(self, block, layers):
        self.inplanes = 64
        super(RefineNetV2_1024, self).__init__()
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
        pretrained_dict = resnet_zoo.load_url(resnet_urls[resnet])
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


class RefineNetV1_1024(nn.Module):
    def __init__(self, block, layers):
        self.inplanes = 64
        super(RefineNetV1_1024, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])        # 256*256*256
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 512*128*128
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # 1024*64*64
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2) # 2048*32*32

        self.middle = nn.Sequential(*make_conv_bn_relu(2048, 1024))     # 1024*32*32

        self.refinenet3 = RCU(1024, 512)       # 512*64*64
        self.trans3 = nn.Sequential(*make_conv_bn_relu(1024, 512))

        self.refinenet2 = RCU(512, 256)        # 256*128*128
        self.trans2 = nn.Sequential(*make_conv_bn_relu(512, 256))

        self.refinenet1 = RCU(256, 64)         # 64*256*256
        self.trans1 = nn.Sequential(*make_conv_bn_relu(256, 64))

        self.refinenet0 = RCU(64, 64)          # 64*512*512
        self.trans0 = nn.Sequential(*make_conv_bn_relu(64, 64))

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

    def load_params(self):
        pretrained_dict = resnet_zoo.load_url(resnet_urls['resnet50'])
        model_dict = self.state_dict()
        model_dict.update({key: pretrained_dict[key] for key in pretrained_dict.keys() if 'fc' not in key})
        self.load_state_dict(model_dict)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x4 = self.middle(x4)

        out = self.refinenet3(x4)
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = out + self.trans3(x3)

        out = self.refinenet2(out)
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = out + self.trans2(x2)

        out = self.refinenet1(out)
        out = F.upsample(out, scale_factor=2, mode='bilinear')
        out = out + self.trans1(x1)

        out = self.refinenet0(out)
        out = out + self.trans0(x)
        out = F.upsample(out, scale_factor=2, mode='bilinear')

        out = self.final(out)

        return out, F.sigmoid(out)

if __name__ == '__main__':
    # from torch.autograd import Variable
    # a = Variable(torch.randn((6, 3, 1024, 1024))).cuda()
    # resnet = RefineNetV4_1024(BasicBlock, [3, 4, 6, 3]).cuda()
    # resnet.load_params('resnet34')
    # resnet = nn.DataParallel(resnet)
    # # resnet.cuda()
    # # print(resnet(a))
    # print(resnet(a))
    # print(resnet)
    # features = make_layers(cfg=cfg['D'], batch_norm=True).modules()
    # for f in features:
    #     print(f)
    a = Variable(torch.randn((4, 3, 1024, 1024))).cuda()
    net = RefineNetV5_1024()
    net.load_vgg16()
    net = nn.DataParallel(net).cuda()
    print(net(a))

    # model_dict = net.state_dict()
    # for key in net.state_dict().keys():
    #     print(key)
    #
    # pretrain_dict = vgg_zoo.load_url(vgg_urls['vgg16_bn'])
    # for key in vgg_zoo.load_url(vgg_urls['vgg16_bn']).keys():
    #     print(key)
    #
    # model_dict.update({key: pretrain_dict[pretrain_key] for key, pretrain_key in zip(model_dict.keys(), pretrain_dict.keys()) if 'classifier' not in pretrain_key})
    #
    # for net_key, pre_key in zip(model_dict.keys(), pretrain_dict.keys()):
    #     print(net_key, pre_key, torch.equal(model_dict[net_key], pretrain_dict[pre_key]))
