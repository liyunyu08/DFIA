import torch
import torch.nn as nn
from models.FSBI import FuseBlock
from kymatio.torch import Scattering2D
import torch.nn.functional as F


class LayerNorm(nn.Module):  # layernorm, but done in the channel dimension #1
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))
        self.b = nn.Parameter(torch.zeros(1, dim,  1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.g + self.b


class LFA(nn.Module):
    def __init__(self, in_channels):
        super(LFA, self).__init__()
        self.conv1 = nn.Conv2d(2*in_channels, in_channels // 4, kernel_size=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels // 4, in_channels, kernel_size=3, padding=1, bias=False)

    def forward(self, wst_low, cnn_spatial):

        fusion = torch.cat([wst_low, cnn_spatial], dim=1)
        attn = F.relu(self.conv1(fusion))
        attn = torch.sigmoid(self.conv2(attn))
        enhanced_feature = attn * cnn_spatial + wst_low
        return enhanced_feature

class Scattering(nn.Module):
    def __init__(self, H,W,J,channels,L=6):
        super(Scattering, self).__init__()
        self.scat = Scattering2D(J=J, L=L,shape=(H, W))
        self.num_coeffs = int(1 + L * J + L * L * J * (J - 1) / 2)
        self.in_dim = self.num_coeffs-1
        self.out_dim = self.in_dim // 2
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
        #
        self.cca_low = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )

    def forward(self, x):
        x = torch.mean(x,dim=1,keepdims=True)
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




def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None,weatherpool=True):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.LeakyReLU(0.1)

        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv3x3(planes, planes)
        self.bn3 = nn.BatchNorm2d(planes)
        self.maxpool = nn.MaxPool2d(stride)
        self.downsample = downsample
        self.stride = stride
        self.weatherpool = weatherpool



    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)


        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        if self.weatherpool:
            out = self.maxpool(out)
        return out


class ResNet(nn.Module):

    def __init__(self,block=BasicBlock):
        self.inplanes = 3
        super(ResNet, self).__init__()

        self.layer1 = self._make_layer(block, 64,  stride=2)
        self.layer2 = self._make_layer(block, 128, stride=2)
        self.layer3 = self._make_layer(block, 256, stride=2)
        self.layer4 = self._make_layer(block, 512, stride=2,weatherpool=False)


        self.scat = Scattering(H=84,W=84,J=2,channels=128)
        self.sp1 = LFA(128)
        self.fuseblock = FuseBlock(128)

        self.scat2 = Scattering(H=21, W=21,J=1,channels=512)
        self.sp2 = LFA(512)
        self.fuseblock2 = FuseBlock(512)
        self.cca1 = nn.Sequential(
            nn.Conv2d(128, 512, kernel_size=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU()
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block, planes, stride,weatherpool=True):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )


        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample,weatherpool))
        self.inplanes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x,Re):
        scat_low, scat_high = self.scat(x)

        x0 = self.layer1(x)
        x1 = self.layer2(x0)

        if Re:
            x1 = self.sp1(scat_low, x1)
            x1 = self.fuseblock(x1, scat_high)


        x2 = self.layer3(x1)
        x3 = self.layer4(x2)

        if Re:
            scat_low_1, scat_high_1 = self.scat2(x1)
            x3 = self.sp2(scat_low_1, x3)
            x3 = self.fuseblock2(x3, scat_high_1)


        return self.cca1(x1),x3

def ResNet12():
    model = ResNet()
    return model






