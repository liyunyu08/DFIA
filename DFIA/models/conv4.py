
import math
import torch
import torch.nn as nn
from models.FSBI import FuseBlock
from kymatio.torch import Scattering2D
import torch.nn.functional as F

def init_layer(L):
    # Initialization using fan-in
    if isinstance(L, nn.Conv2d):
        n = L.kernel_size[0] * L.kernel_size[1] * L.out_channels
        L.weight.data.normal_(0, math.sqrt(2.0 / float(n)))
    elif isinstance(L, nn.BatchNorm2d):
        L.weight.data.fill_(1)
        L.bias.data.fill_(0)


class LFA(nn.Module):
    def __init__(self, in_channels):
        super(LFA, self).__init__()
        self.conv1 = nn.Conv2d(2*in_channels, in_channels // 2, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, wst_low, cnn_spatial):

        fusion = torch.cat([wst_low, cnn_spatial], dim=1)
        attn = F.relu(self.conv1(fusion))
        attn = torch.sigmoid(self.conv2(attn))
        enhanced_feature = attn * cnn_spatial + wst_low
        return enhanced_feature



class Scattering(nn.Module):
    def __init__(self, H,W,J,channels,L=8):
        super(Scattering, self).__init__()
        self.scat = Scattering2D(J=J, L=L,shape=(H, W))
        self.num_coeffs = int(1 + L * J + L * L * J * (J - 1) / 2)
        self.in_dim = self.num_coeffs-1
        self.out_dim = self.in_dim //2
        self.in_dim_low = 1

        self.scattering2dConv2_low = nn.Sequential(
                                  nn.Conv2d(self.in_dim_low, self.in_dim_low, 3, 1, 1),
                                  nn.BatchNorm2d(self.in_dim_low),
                                  nn.ReLU(True),
                                  nn.Conv2d(self.in_dim_low, self.in_dim_low, 3, 1, 1),
                                  nn.BatchNorm2d(self.in_dim_low),
                                  nn.ReLU(True))

        self.scattering2dConv2_high = nn.Sequential(
                                  nn.Conv2d(self.in_dim, self.out_dim, 3, 1, 1),
                                  nn.BatchNorm2d(self.out_dim),
                                  nn.ReLU(True),
                                  nn.Conv2d(self.out_dim, self.in_dim, 3, 1, 1),
                                  nn.BatchNorm2d(self.in_dim),
                                  nn.ReLU(True))

        self.cca_high = nn.Sequential(
            nn.Conv2d(self.in_dim, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

        self.cca_low = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.mean(x, dim=1, keepdims=True)
        scat_feat = self.scat(x)
        b,c,n,h,w = scat_feat.shape

        scat_low = scat_feat[:, :, 0, :, :]
        scat_low = scat_low.contiguous().view(b, -1, h, w)

        scat_low = self.scattering2dConv2_low(scat_low)
        scat_low = self.cca_low(scat_low)

        scat_high = scat_feat[:, :, 1:, :, :]
        scat_high = scat_high.contiguous().view(b, -1, h, w)
        scat_high = self.scattering2dConv2_high(scat_high)
        scat_high = self.cca_high(scat_high)
        return scat_low,scat_high



class ConvBlock(nn.Module):
    maml = False  # Default

    def __init__(self, indim, outdim, pool=True, padding=1):
        super(ConvBlock, self).__init__()
        self.indim = indim
        self.outdim = outdim

        self.C = nn.Conv2d(indim, outdim, 3, padding=padding)
        self.BN = nn.BatchNorm2d(outdim)
        self.relu = nn.ReLU(inplace=True)

        self.parametrized_layers = [self.C, self.BN, self.relu]
        if pool:
            self.pool = nn.MaxPool2d(2)
            self.parametrized_layers.append(self.pool)

        for layer in self.parametrized_layers:
            init_layer(layer)

        self.trunk = nn.Sequential(*self.parametrized_layers)

    def forward(self, x):
        out = self.trunk(x)
        return out



class ConvNet(nn.Module):
    def __init__(self,depth=4):
        super(ConvNet, self).__init__()
        trunk = []
        for i in range(depth):
            indim = 3 if i == 0 else 64
            outdim = 64
            B = ConvBlock(indim, outdim, pool=(i <3))  # only pooling for fist 4 layers
            trunk.append(B)

        self.trunk = nn.Sequential(*trunk)
        self.fuseblock = FuseBlock(64)
        self.scat = Scattering(H=84,W=84,J=2,channels=64)
        self.scat2 = Scattering(H=21, W=21,J=1,L=8,channels=64)
        self.sp = LFA(64)

    def forward(self, spac,Re=True):

        out_0 = self.trunk[0](spac)
        out_1 = self.trunk[1](out_0)

        if Re:
            scat_low, scat_high = self.scat(spac)
            out_1 = self.sp(scat_low,out_1)
            out_1 = self.fuseblock(out_1, scat_high)

        out_2 = self.trunk[2](out_1)
        out_3 = self.trunk[3](out_2)

        if Re:
            scat_low_1, scat_high_1 = self.scat2(out_1)
            out_3 = self.sp(scat_low_1, out_3)
            out_3 = self.fuseblock(out_3, scat_high_1)


        return out_1,out_3


def Conv4():
    return ConvNet(4)


