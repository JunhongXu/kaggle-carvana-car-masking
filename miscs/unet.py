import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class UNetBlockBn(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(UNetBlockBn, self).__init__()
        self.conv1 = nn.Conv2d(in_feats, out_channels=out_feats, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_feats)
        self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_feats)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        return x


class UNetUpBlockBn(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(UNetUpBlockBn, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_feats, out_feats*2, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_feats*2)
        # half the channel
        self.conv2 = nn.Conv2d(out_feats*2, out_feats, kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_feats)

    def forward(self, x, feat):
        # upsampling
        x = self.upsampling(x)
        x = torch.cat((x, feat), 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return x


class UNetBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_feats, out_channels=out_feats, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_feats)
        self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1)
        # self.bn2 = nn.BatchNorm2d(out_feats)

    def forward(self, x):
        x = self.conv1(x)
        # x = self.bn2(x)
        x = F.relu(x)

        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_feats, out_feats, out_size=None):
        super(UNetUpBlock, self).__init__()
        self.out_size = out_size
        self.upsampling = nn.Upsample(scale_factor=2)
        self.conv1 = nn.Conv2d(in_feats, out_feats*2, kernel_size=3, padding=1)
        # self.bn1 = nn.BatchNorm2d(out_feats*2)
        # self.conv2 = nn.Conv2d(out_feats*2, out_feats, kernel_size=3, padding=1, bias=False)
        # self.bn2 = nn.BatchNorm2d(out_feats)
        # half the channel
        self.conv2 = nn.Conv2d(out_feats*2, out_feats, kernel_size=1, stride=1)
        # self.bn2 = nn.BatchNorm2d(out_feats)

    def forward(self, x, feat):
        # upsampling
        x = self.upsampling(x)
        # feat = self.crop(feat)
        x = torch.cat((x, feat), 1)
        x = self.conv1(x)
        # x = self.bn1(x)
        x = F.relu(x)
        x = self.conv2(x)
        # x = self.bn2(x)
        x = F.relu(x)
        # x = self.conv3(x)
        # x = self.bn3(x)
        # x = F.relu(x)
        return x


class UNet512(nn.Module):
    def __init__(self):
        super(UNet512, self).__init__()
        self.downblock1 = UNetBlockBn(3, 32)      # 32*512*512
        self.pool1 = nn.MaxPool2d(2, stride=2)   # 32*256*256

        self.downblock2 = UNetBlockBn(32, 64)    # 64*256*256
        self.pool2 = nn.MaxPool2d(2, stride=2)   # 64*128*128

        self.downblock3 = UNetBlockBn(64, 128)    # 128*128*128
        self.pool3 = nn.MaxPool2d(2, 2)         # 128*64*64

        self.downblock4 = UNetBlockBn(128, 256)   # 256*64*64
        self.pool4 = nn.MaxPool2d(2, 2)         # 256*32*32

        self.downblock5 = UNetBlockBn(256, 512)     # 512*32*32
        self.pool5 = nn.MaxPool2d(2, 2)             # 512*16*16

        # transition block
        self.transition = nn.Sequential(
            nn.Conv2d(512, 1024, 1, bias=False),    # 1024*16*16
            nn.BatchNorm2d(1024),
            nn.ReLU(),

            nn.Conv2d(1024, 512, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        self.upblock1 = UNetUpBlockBn(1024, 256)     # 256*32*32
        self.upblock2 = UNetUpBlockBn(512, 128)        # 128*64*64
        self.upblock3 = UNetUpBlockBn(256, 64)          # 64*128*128
        self.upblock4 = UNetUpBlockBn(128, 32)         # 32*256*256
        self.upblock5 = UNetUpBlockBn(64, 32)       # 16*512*512
        self.upblock6 = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels=32, out_channels=16, padding=0, kernel_size=1, bias=False),
            nn.BatchNorm2d(16)
        )

        self.final_conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(16, 1, 1)
        )

    def forward(self, x):
        feat1 = self.downblock1(x)
        out = self.pool1(feat1)
        feat2 = self.downblock2(out)
        out = self.pool2(feat2)   # 64*64*96
        feat3 = self.downblock3(out)   # 128*32*48
        out = self.pool3(feat3)   # 128*32*32
        feat4 = self.downblock4(out)   # 256*16*24
        out = self.pool4(feat4)   # 256*16*24
        feat5 = self.downblock5(out)
        out = self.pool5(feat5)
        out = self.transition(out)
        up1 = self.upblock1(out, feat5)
        up2 = self.upblock2(up1, feat4)
        up3 = self.upblock3(up2, feat3)
        up4 = self.upblock4(up3, feat2)
        up5 = self.upblock5(up4, feat1)
        up6 = self.upblock6(up5)
        logits = self.final_conv(up6)
        probs = F.sigmoid(logits)
        return logits, probs


class UNetV2(nn.Module):
    def __init__(self):
        super(UNetV2, self).__init__()
        self.downblock1 = UNetBlock(3, 64)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 32*128*128

        self.downblock2 = UNetBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 64*64*64

        self.downblock3 = UNetBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2, 2)         # 128*32*32

        self.downblock4 = UNetBlock(256, 512)
        self.pool4 = nn.MaxPool2d(2, 2)         # 256*16*16

        self.downblock5 = UNetBlock(512, 1024)   # 512*16*16
        self.downblock6 = nn.Conv2d(1024, 512, 1)    #256*16*16

        self.upblock1 = UNetUpBlockBn(1024, 256)   # 256*32*32
        self.upblock2 = UNetUpBlock(768, 256, 64)   # 128*64*64
        self.upblock3 = UNetUpBlock(384, 128, 128)    # 64*128*128
        self.upblock4 = UNetUpBlock(192, 64, 256)    # 32*256*256

        self.final_conv = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 2, 1)
        )

    def forward(self, x):
        feat1 = self.downblock1(x)       # 32*128*192
        out = self.pool1(feat1)   # 32*128*192

        feat2 = self.downblock2(out)   # 64*64*96
        out = self.pool2(feat2)   # 64*64*96

        feat3 = self.downblock3(out)   # 128*32*48
        out = self.pool3(feat3)   # 128*32*32

        feat4 = self.downblock4(out)   # 256*16*24
        out = self.pool4(feat4)   # 256*16*24

        out = self.downblock5(out)    # 512*16*24
        out = self.downblock6(out)      # 256*16*24

        up1 = self.upblock1(out, feat4) # 256*32*48
        up2 = self.upblock2(up1, feat3) # 64*96
        up3 = self.upblock3(up2, feat2) # 128*192
        up4 = self.upblock4(up3, feat1) # 256*384

        logits = self.final_conv(up4)
        probs = F.log_softmax(logits)

        return logits, probs


class UNetV3(nn.Module):    # small one with bn
    def __init__(self):
        super(UNetV3, self).__init__()
        self.downblock1 = UNetBlock(3, 16)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 32*128*128

        self.downblock2 = UNetBlock(16, 32)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 64*64*64

        self.downblock3 = UNetBlock(32, 64)
        self.pool3 = nn.MaxPool2d(2, 2)         # 128*32*32

        self.downblock4 = UNetBlock(64, 128)
        self.pool4 = nn.MaxPool2d(2, 2)         # 256*16*16

        self.downblock5 = UNetBlock(128, 256)   # 512*16*16
        self.downblock6 = nn.Conv2d(256, 128, 1)    #256*16*16
        # self.bn1 = nn.BatchNorm2d(128)

        self.upblock1 = UNetUpBlock(256, 128, 32)   # 256*32*32
        self.upblock2 = UNetUpBlock(192, 64, 64)   # 128*64*64
        self.upblock3 = UNetUpBlock(96, 32, 128)    # 64*128*128
        self.upblock4 = UNetUpBlock(48, 16, 256)    # 32*256*256

        self.final_conv = nn.Sequential(
            nn.Conv2d(16, 16, 1),
            # nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 2, 1)
        )

    def forward(self, x):
        feat1 = self.downblock1(x)       # 32*128*192
        out = self.pool1(feat1)   # 32*128*192

        feat2 = self.downblock2(out)   # 64*64*96
        out = self.pool2(feat2)   # 64*64*96

        feat3 = self.downblock3(out)   # 128*32*48
        out = self.pool3(feat3)   # 128*32*32

        feat4 = self.downblock4(out)   # 256*16*24
        out = self.pool4(feat4)   # 256*16*24

        out = self.downblock5(out)    # 512*16*24
        out = self.downblock6(out)      # 256*16*24
        # out = self.bn1(out)
        out = F.relu(out)

        up1 = self.upblock1(out, feat4) # 256*32*48
        up2 = self.upblock2(up1, feat3) # 64*96
        up3 = self.upblock3(up2, feat2) # 128*192
        up4 = self.upblock4(up3, feat1) # 256*384

        logits = self.final_conv(up4)
        probs = F.log_softmax(logits)

        return logits, probs

if __name__ == '__main__':
    loss = nn.NLLLoss2d()
    _x = Variable(torch.randn(1, 3, 512, 512))
    target = Variable(torch.LongTensor(1, 512, 512).random_(0, 1))
    net = UNet512()
    logits, probs = net(_x)
    print(loss(probs, target))

