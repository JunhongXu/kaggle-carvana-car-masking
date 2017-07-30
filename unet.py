import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


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
        # ¸¸¸x = self.bn2(x)
        x = F.relu(x)

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_feats, out_feats, out_size):
        super(UNetUpBlock, self).__init__()
        self.out_size = out_size
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1)
       # self.bn1 = nn.BatchNorm2d(out_feats)
        self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1)
       # self.bn2 = nn.BatchNorm2d(out_feats)

    def crop(self, feat):
        """
        :param feat: N * C * H * W
        :return: cropped feature with shape N * C * out_size * out_size
        """
        feat_size = feat.size(-1)
        cropped_size = (feat_size - self.out_size) // 2
        feat = feat[:, :, cropped_size:(feat_size-cropped_size), cropped_size:(feat_size-cropped_size)]
        return feat

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
        return x


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        self.downblock1 = UNetBlock(3, 32)
        self.pool1 = nn.MaxPool2d(2, stride=2)  # 32*128*128

        self.downblock2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2, stride=2)  # 64*64*64

        self.downblock3 = UNetBlock(64, 128)
        self.pool3 = nn.MaxPool2d(2, 2)         # 128*32*32

        self.downblock4 = UNetBlock(128, 256)
        self.pool4 = nn.MaxPool2d(2, 2)         # 256*16*16

        self.downblock5 = UNetBlock(256, 512)   # 512*16*16
        self.downblock6 = nn.Conv2d(512, 256, 1)    #256*16*16

        self.upblock1 = UNetUpBlock(512, 256, 32)   # 256*32*32
        self.upblock2 = UNetUpBlock(384, 128, 64)   # 128*64*64
        self.upblock3 = UNetUpBlock(192, 64, 128)    # 64*128*128
        self.upblock4 = UNetUpBlock(96, 64, 256)    # 32*256*256

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


if __name__ == '__main__':
    loss = nn.NLLLoss2d()
    _x = Variable(torch.randn(1, 3, 256, int(256*1.5)))
    target = Variable(torch.LongTensor(1, 256, int(256*1.5)).random_(0, 1))
    net = UNet()
    logits, probs = net(_x)
    print(loss(probs, target))

