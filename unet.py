import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.autograd import Variable


class UNetBlock(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(UNetBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_feats, out_channels=out_feats, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        return x


class UNetUpBlock(nn.Module):
    def __init__(self, in_feats, out_feats, out_size):
        super(UNetUpBlock, self).__init__()
        self.out_size = out_size
        self.upsampling = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv1 = nn.Conv2d(in_feats, out_feats, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_feats, out_feats, kernel_size=3, padding=1)

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
        x = F.relu(x)
        x = self.conv2(x)
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

        self.upblock1 = UNetUpBlock(384, 128, 32)   # 256*32*32
        self.upblock2 = UNetUpBlock(192, 64, 64)   # 128*64*64
        self.upblock3 = UNetUpBlock(96, 32, 128)    # 64*128*128
        # self.upblock4 = UNetUpBlock(32, 32, 256)    # 32*256*256

        self.final_conv = nn.Conv2d(32, 2, 1)

    def forward(self, x):
        feat1 = self.downblock1(x)
        feat1 = self.pool1(feat1)   # 32*128*128

        feat2 = self.downblock2(feat1)
        feat2 = self.pool2(feat2)   # 64*64*64

        feat3 = self.downblock3(feat2)
        feat3 = self.pool3(feat3)   # 128*32*32

        feat4 = self.downblock4(feat3)
        feat4 = self.pool4(feat4)   # 256*16*16

        out = self.downblock5(feat4)    # 512*16*16
        out = self.downblock6(out)      # 256*16*16

        up1 = self.upblock1(out, feat3) # 256*32*32
        up2 = self.upblock2(up1, feat2)
        up3 = self.upblock3(up2, feat1)
        # up4 = self.upblock4(up3, feat1)

        logits = self.final_conv(up3)
        probs = F.softmax(logits)

        return logits, probs


if __name__ == '__main__':
    x = Variable(torch.randn(1, 3, 256, 256))
    net = UNet()
    print(net(x)[1])

