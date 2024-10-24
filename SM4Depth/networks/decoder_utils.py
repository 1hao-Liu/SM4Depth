import math
import torch
import torch.nn.functional as F
import torch.nn as nn


class SkipConnect(nn.Module):
    def __init__(self, num_features=[], decoder_feature=[]):
        super().__init__()
        self.conv4 = nn.Conv2d(num_features[-1], decoder_feature[-1], kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(num_features[-2], decoder_feature[-2], kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(num_features[-3], decoder_feature[-3], kernel_size=3, stride=1, padding=1)
        self.conv1 = nn.Conv2d(num_features[-4], decoder_feature[-4], kernel_size=3, stride=1, padding=1)


class ResidualConvUnit(nn.Module):
    def __init__(self, features_channels):
        super().__init__()
        self.activation = nn.LeakyReLU(False)
        self.skip_add = nn.quantized.FloatFunctional()
        self.conv1 = nn.Conv2d(features_channels, features_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(features_channels, features_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(features_channels)
        self.bn2 = nn.BatchNorm2d(features_channels)


    def forward(self, x):
        out = self.activation(x)
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.skip_add.add(out, x)

        return out

class ResRefineModule(nn.Module):
    def __init__(self, in_channels, up_channels):
        super().__init__()
        self.conv_out = nn.Conv2d(in_channels, 256, kernel_size=3, stride=1, padding=1)
        self.rrm = nn.Sequential(nn.Conv2d(in_channels, up_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(up_channels),
                                 nn.LeakyReLU(False),
                                 nn.Conv2d(up_channels, up_channels, kernel_size=3, stride=1, padding=1, bias=False),
                                 nn.BatchNorm2d(up_channels),
                                 nn.LeakyReLU(False),
                                 nn.Conv2d(up_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=False))

    def forward(self, probmap, depthmap):
        depthmap = upsample(depthmap, probmap.size(2), probmap.size(3), "bilinear")  #[10, 1, 53, 71]
        probmap = torch.cat((probmap, depthmap), dim=1) # [10, 97, 27, 36]
        out = self.rrm(probmap) # [10, 97, 27, 36]
        out = self.conv_out(out + probmap) # [10, 256, 27, 36]

        return out


def PredictHead(probMap, bins):
    probMap = F.softmax(probMap, dim=1)
    depthmap = torch.sum(probMap * bins, dim=1, keepdim=True)

    return depthmap


def upsample(x, up_h, up_w, mode="bilinear"):
    if mode == "bilinear":
        return F.interpolate(x, size=[up_h, up_w], mode="bilinear", align_corners=True)
    else:
        return F.interpolate(x, size=[up_h, up_w], mode="nearest")


class Refine_Decoder(nn.Module):
    def __init__(self, in_channels, up_channels, out_channels, up_mode):
        super().__init__()
        self.resConfUnit1 = ResidualConvUnit(in_channels)
        self.resConfUnit2 = ResidualConvUnit(in_channels)   # 巩固一下特征
        self.RRM = ResRefineModule(in_channels + 1, up_channels) # (97,192)
        self.skip_add = nn.quantized.FloatFunctional()
        self.conv_out = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv_f = nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1)
        self.up_mode = up_mode

    def forward(self, *xs):
        x = xs[0]  # decoder input
        if len(xs) == 6:  # skip input
            y = self.resConfUnit1(xs[1]) #[10, 96, 53, 71]
            x = self.skip_add.add(x, y)   #[10， 96， 53，71]
        up_h, up_w, bins = xs[-3], xs[-2], xs[-1]

        x = self.resConfUnit2(x)
        if len(xs) == 6:
            rrm_depthmap = self.RRM(x, xs[2])
            depthmap = PredictHead(rrm_depthmap, bins)
        else:
            probmap = self.conv_f(x)
            depthmap = PredictHead(probmap, bins)

        x = upsample(x, up_h=up_h, up_w=up_w)
        x = self.conv_out(x)

        return x, depthmap


class HSC_Decoder(nn.Module):  
    def __init__(self, num_features, skip_channels, up_mode):
        super().__init__()
        self.skip_connect = SkipConnect(num_features, skip_channels)
        self.refinenet4 = Refine_Decoder(in_channels=384, up_channels=768, out_channels=192, up_mode=up_mode)
        self.refinenet3 = Refine_Decoder(in_channels=192, up_channels=384, out_channels=96,  up_mode=up_mode)
        self.refinenet2 = Refine_Decoder(in_channels=96,  up_channels=192, out_channels=48,  up_mode=up_mode)
        self.refinenet1 = Refine_Decoder(in_channels=48,  up_channels=96,  out_channels=16,  up_mode=up_mode)
        self.refine_out = ResRefineModule(17, 36)
        self.up_mode = up_mode

    def forward(self, f1, f2, f3, f4, bins):
        f4 = self.skip_connect.conv4(f4)
        f4_refine, r4 = self.refinenet4(f4, 27, 36, bins)

        f3 = self.skip_connect.conv3(f3)
        f3_refine, r3 = self.refinenet3(f4_refine, f3, r4,  53,  71, bins)
        
        f2 = self.skip_connect.conv2(f2)
        f2_refine, r2 = self.refinenet2(f3_refine, f2, r3, 106, 141, bins)

        f1 = self.skip_connect.conv1(f1)
        f1_refine, r1 = self.refinenet1(f2_refine, f1, r2, 212, 282, bins)

        out = self.refine_out(f1_refine, r1)
        out = PredictHead(out, bins)
        out = upsample(out, 424, 564, "bilinear")

        return [r3, r2, r1, out]
