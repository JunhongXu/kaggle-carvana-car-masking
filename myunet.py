# unet from scratch


import torch
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
#  https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
#  https://discuss.pytorch.org/t/solved-what-is-the-correct-way-to-implement-custom-loss-function/3568/4
class CrossEntropyLoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(CrossEntropyLoss2d, self).__init__()
        self.nll_loss = nn.NLLLoss2d(weight, size_average)

    def forward(self, logits, targets):
        return self.nll_loss(F.log_softmax(logits), targets)

#https://github.com/bermanmaxim/jaccardSegment/blob/master/losses.py
class StableBCELoss(nn.modules.Module):
       def __init__(self):
             super(StableBCELoss, self).__init__()
       def forward(self, input, target):
             neg_abs = - input.abs()
             loss = input.clamp(min=0) - input * target + (1 + neg_abs.exp()).log()
             return loss.mean()

class BCELoss2d(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BCELoss2d, self).__init__()
        self.bce_loss = nn.BCELoss(weight, size_average)

    def forward(self, logits, targets, weight=None):
        w = weight.view(-1)
        z = logits.view(-1)
        t = targets.view(-1)
        loss = w*z.clamp(min=0) - w*z*t + w*torch.log(1 + torch.exp(-z.abs()))
        loss = loss.sum()/w.sum()
        return loss

        #
        # logits_flat  = logits.view (-1)
        # targets_flat = targets.view(-1)
        # return StableBCELoss()(logits_flat,targets_flat)


class SoftDiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(SoftDiceLoss, self).__init__()

    def forward(self, logits, targets):
        num = targets.size(0)
        probs = F.sigmoid(logits)
        m1  = probs.view(num,-1)
        m2  = targets.view(num,-1)
        intersection = (m1 * m2)

        score = 2. * (intersection.sum(1)+1) / (m1.sum(1) + m2.sum(1)+1)
        score = 1- score.sum()/num
        return score


class SoftIoULoss(nn.Module):
    def __init__(self):
        super(SoftIoULoss, self).__init__()

    def forward(self, logits, targets, weight=None):
        batch_size = targets.size(0)
        probs = F.sigmoid(logits)
        probs = probs.view(batch_size, -1)
        targets = targets.view(batch_size, -1)
        intersection = torch.sum(probs * targets)
        union = torch.sum((probs + targets) - (probs * targets))
        iou = intersection / union
        loss = 1 - iou
        return loss


def make_conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, padding=1):
    return [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


class UNet128 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet128, self).__init__()
        in_channels, height, width = in_shape

        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=1, stride=1, padding=0 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=1, stride=1, padding=0 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #128

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)
        out   = self.classify(out)
        #out   = F.sigmoid(out)

        return out



# a bigger version for 256x256
class UNet256 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256, self).__init__()
        in_channels, height, width = in_shape

        #256

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=1, stride=2, padding=0 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=1, stride=1, padding=0 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=1, stride=1, padding=0 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=1, stride=1, padding=0 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #256

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = self.up0(out)

        out   = self.classify(out)

        return out





# a bigger version for 256x256
class UNet256_1 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256_1, self).__init__()
        in_channels, height, width = in_shape

        #256

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=2, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #256

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = self.up0(out)

        out   = self.classify(out)

        return out






class UNet128_1 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet128_1, self).__init__()
        in_channels, height, width = in_shape

        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(32, 64,  kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 512, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.same = nn.Sequential(
            *make_conv_bn_relu(512,512, kernel_size=1, stride=1, padding=0 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu( 512,128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.same(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = self.classify(out)

        return out



# based on https://github.com/jocicmarko/ultrasound-nerve-segmentation/blob/master/train.py

class UNet128_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet128_2, self).__init__()
        in_channels, height, width = in_shape

        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(64, num_classes, kernel_size=1, stride=1, padding=0 )



    def forward(self, x):

        #256

        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = self.classify(out)

        return out


class UNet256_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet256_2, self).__init__()
        in_channels, height, width = in_shape

        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(32, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #256
        down0 = self.down0(x)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)

        out   = self.classify(out)

        return out



class UNet512_2 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet512_2, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down0a = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down0 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down1 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down2 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up4 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up3 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up2 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up0a = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(16, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down0a = self.down0a(x)
        out    = F.max_pool2d(down0a, kernel_size=2, stride=2) #64

        down0 = self.down0(out)
        out   = F.max_pool2d(down0, kernel_size=2, stride=2) #64

        down1 = self.down1(out)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #32

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #16

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down0, out],1)
        out   = self.up0(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down0a, out],1)
        out   = self.up0a(out)

        out   = self.classify(out)

        return out

class UNet_1024_5 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet_1024_5, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(  3+16,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #64

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #64

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #32

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #16

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        # out   = F.upsample_bilinear(out, scale_factor=2) #1024
        # print(out.size())
        out   = torch.cat([x, out],1)
        out   = self.up0(out)


        out   = self.classify(out)
        logtis = F.sigmoid(out)

        return out, logtis


class UNet_double_1024_5 (nn.Module):

    def __init__(self, in_shape, num_classes):
        super(UNet_double_1024_5, self).__init__()
        in_channels, height, width = in_shape


        #512
        self.down1 = nn.Sequential(
            *make_conv_bn_relu(in_channels, 16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(16, 16, kernel_size=3, stride=1, padding=1 ),
        )
        #256


        #UNet512_2 ------------------------------------------------------------------------
        #256
        self.down2 = nn.Sequential(
            *make_conv_bn_relu(16, 32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(32, 32, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.down3 = nn.Sequential(
            *make_conv_bn_relu(32, 64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(64, 64, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.down4 = nn.Sequential(
            *make_conv_bn_relu(64,  128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(128, 128, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.down5 = nn.Sequential(
            *make_conv_bn_relu(128, 256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(256, 256, kernel_size=3, stride=1, padding=1 ),
        )
        #16

        self.down6 = nn.Sequential(
            *make_conv_bn_relu(256,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(512,512, kernel_size=3, stride=1, padding=1 ),
        )
        #8

        self.center = nn.Sequential(
            *make_conv_bn_relu(512, 1024, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(1024,1024, kernel_size=3, stride=1, padding=1 ),
        )

        #16
        self.up6 = nn.Sequential(
            *make_conv_bn_relu(512+1024,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     512,512, kernel_size=3, stride=1, padding=1 ),
            #nn.Dropout(p=0.10),
        )
        #16

        self.up5 = nn.Sequential(
            *make_conv_bn_relu(256+512,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    256,256, kernel_size=3, stride=1, padding=1 ),
        )
        #32

        self.up4 = nn.Sequential(
            *make_conv_bn_relu(128+256,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    128,128, kernel_size=3, stride=1, padding=1 ),
        )
        #64

        self.up3 = nn.Sequential(
            *make_conv_bn_relu( 64+128,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     64,64, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up2 = nn.Sequential(
            *make_conv_bn_relu( 32+64,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    32,32, kernel_size=3, stride=1, padding=1 ),
        )
        #128
        #-------------------------------------------------------------------------

        self.up1 = nn.Sequential(
            *make_conv_bn_relu( 16+32,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(    16,16, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.up0 = nn.Sequential(
            *make_conv_bn_relu(  3+16,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
            *make_conv_bn_relu(     8,8, kernel_size=3, stride=1, padding=1 ),
        )
        #128

        self.classify = nn.Conv2d(8, num_classes, kernel_size=1, stride=1, padding=0 )


    def forward(self, x):

        #512
        down1 = self.down1(x)
        out   = F.max_pool2d(down1, kernel_size=2, stride=2) #64

        down2 = self.down2(out)
        out   = F.max_pool2d(down2, kernel_size=2, stride=2) #64

        down3 = self.down3(out)
        out   = F.max_pool2d(down3, kernel_size=2, stride=2) #64

        down4 = self.down4(out)
        out   = F.max_pool2d(down4, kernel_size=2, stride=2) #32

        down5 = self.down5(out)
        out   = F.max_pool2d(down5, kernel_size=2, stride=2) #16

        down6 = self.down6(out)
        out   = F.max_pool2d(down6, kernel_size=2, stride=2) # 8

        out   = self.center(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #16
        out   = torch.cat([down6, out],1)
        out   = self.up6(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #32
        out   = torch.cat([down5, out],1)
        out   = self.up5(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #64
        out   = torch.cat([down4, out],1)
        out   = self.up4(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down3, out],1)
        out   = self.up3(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #128
        out   = torch.cat([down2, out],1)
        out   = self.up2(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #256
        out   = torch.cat([down1, out],1)
        out   = self.up1(out)

        out   = F.upsample_bilinear(out, scale_factor=2) #1024
        x     = F.upsample_bilinear(x,   scale_factor=2)
        out   = torch.cat([x, out],1)
        out   = self.up0(out)


        out   = self.classify(out)
        logtis = F.sigmoid(out)

        return out, logtis


# main #################################################################
if __name__ == '__main__':
    print( '%s: calling main function ... ' % os.path.basename(__file__))
    batch_size = 4
    C,H,W = 3, 1280, 1920

    # if 0: # CrossEntropyLoss2d()
    #     inputs = torch.randn(batch_size,C,H,W)
    #     labels = torch.LongTensor(batch_size,H,W).random_(1)
    #
    #     net = UNet512_2(in_shape=(C,H,W), num_classes=2).cuda().train()
    #     x = Variable(inputs).cuda()
    #     y = Variable(labels).cuda()
    #     logits = net.forward(x)
    #
    #     loss = CrossEntropyLoss2d()(logits, y)
    #     loss.backward()
    #
    #     print(type(net))
    #     print(net)
    #
    #     print('logits')
    #     print(logits)



    if 1: # BCELoss2d()
        # num_classes = 1
        #
        # inputs = torch.randn(batch_size,C,H,W)
        # labels = torch.LongTensor(batch_size,H, W).random_(1).type(torch.FloatTensor)
        #
        # net = UNet_1024_5(in_shape=(C,H,W), num_classes=1).cuda().train()
        # net = nn.DataParallel(net)
        # x = Variable(inputs).cuda()
        # y = Variable(labels).cuda()
        # logits = net.forward(x)
        #
        # loss = BCELoss2d()(logits, y[0])
        # loss.backward()
        #
        # print(type(net))
        # print(net)
        #
        # print('logits')
        # print(logits)
        labels = Variable(torch.FloatTensor(30, 128, 128).random_(2))
        predictions = Variable(torch.randn(30, 128, 128))
        loss = SoftIoULoss()
        print(loss(predictions, labels))
    #input('Press ENTER to continue.')
