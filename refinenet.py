import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision.models.resnet import Bottleneck
from torchvision.models.resnet import model_urls, model_zoo
from torchvision.models.densenet import _DenseBlock, _Transition


def make_conv_bn_relu(input_feat, output_feat, kernel_size=3, padding=1):

    layer = (nn.Conv2d(input_feat, output_feat, kernel_size=kernel_size, stride=1, padding=padding, bias=False),
             nn.BatchNorm2d(output_feat), nn.ReLU())

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


class RefineNetV3_1024(nn.Module):
    def __init__(self, growth_rate=64, block_config=(3, 6, 10, 4),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):
        super(RefineNetV3_1024, self).__init__()
        self.num_features = num_init_features
        self.block_config = block_config
        self.conv1 = nn.Sequential(
            *make_conv_bn_relu(3, num_init_features),
            nn.MaxPool2d(2, 2)
        )                                               # 64*512*512

        self.layer1 = self._make_layer(block_config[0], bn_size, growth_rate, drop_rate, 0) # 128*256*256
        self.layer2 = self._make_layer(block_config[1], bn_size, growth_rate, drop_rate, 1) # 256*128*128
        self.layer3 = self._make_layer(block_config[2], bn_size, growth_rate, drop_rate, 2) # 512*64*64
        self.layer4 = self._make_layer(block_config[0], bn_size, growth_rate, drop_rate, 3) # 352*32*32
        self.norm5 = nn.BatchNorm2d(self.num_features)

        # middle transition
        self.trans4 = nn.Sequential(*make_conv_bn_relu(320, 512, kernel_size=1, padding=0))  # 512*32*32

        # upblock
        self.up_3 = RCU(1024, 512)  # 512*64*64
        self.trans3 = nn.Sequential(RCU(448, 512))

        self.up_2 = RCU(768, 256)   # 256*128*128
        self.trans2 = nn.Sequential(RCU(256, 256))

        self.up_1 = RCU(384, 128)   # 128*256*256
        self.trans1 = nn.Sequential(RCU(128, 128))

        self.up_0 = RCU(192, 64)    # 64*512*512
        self.trans0 = nn.Sequential(RCU(64, 64))

        self.classify = nn.Sequential(
            RCU(64, 32),
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(32, 1, bias=True, padding=1, stride=1, kernel_size=3)
        )

    def _make_layer(self, num_layers, bn_size, growth_rate, drop_rate, block_idx):
        denseblock = _DenseBlock(num_layers, self.num_features, bn_size, growth_rate, drop_rate)
        self.num_features = self.num_features + num_layers * growth_rate
        if block_idx != len(self.block_config):
            trans = _Transition(self.num_features, self.num_features//2)
            self.num_features = self.num_features//2
        return nn.Sequential(denseblock, trans)

    def forward(self, x):
        down1 = self.conv1(x)
        print(down1.size()) # 64
        down2 = self.layer1(down1)
        print(down2.size()) # 128
        down3 = self.layer2(down2)
        print(down3.size()) # 256
        down4 = self.layer3(down3)
        print(down4.size()) # 512
        down5 = self.layer4(down4)
        print(down5.size()) # 384
        down5 = self.norm5(down5)

        middle = self.trans4(down5)
        print(middle.size())
        out = F.upsample(middle, scale_factor=2, mode='bilinear')
        down4 = self.trans3(down4)
        out = torch.cat((out, down4), 1)
        out = self.up_3(out)
        print(out.size())

        out = F.upsample(out, scale_factor=2, mode='bilinear')
        down3 = self.trans2(down3)
        out = torch.cat((out, down3), 1)
        out = self.up_2(out)
        print(out.size())

        out = F.upsample(out, scale_factor=2, mode='bilinear')
        down2 = self.trans1(down2)
        out = torch.cat((out, down2), 1)
        out = self.up_1(out)
        print(out.size())

        out = F.upsample(out, scale_factor=2, mode='bilinear')
        down1 = self.trans0(down1)
        out = torch.cat((out, down1), 1)
        out = self.up_0(out)
        print(out.size())
        out = self.classify(out)
        return out


class RefineNetV2_1024(nn.Module):
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
    from torch.autograd import Variable
    a = Variable(torch.randn((4, 3, 1024, 1024))).cuda()
    resnet = RefineNetV3_1024()
    # resnet.load_params()
    resnet = nn.DataParallel(resnet)
    resnet.cuda()
    # print(resnet(a))
    print(resnet(a))
    print(resnet)
