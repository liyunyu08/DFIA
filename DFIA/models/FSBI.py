
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from einops import rearrange, repeat



class Attention(nn.Module):
    def __init__(self, dim=64, num_heads=8, bias=False):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim , kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x, y):

        b, c, h, w = x.shape

        kv = self.kv_dwconv(self.kv(y))
        k, v = kv.chunk(2, dim=1)
        q = self.q_dwconv(self.q(x))

        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)

        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)

        out = (attn @ v)
        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out = self.project_out(out)
        return out

class FuseBlock(nn.Module):
    def __init__(self, channels):
        super(FuseBlock, self).__init__()
        self.fre = nn.Conv2d(channels, channels, 3, 1, 1)
        self.spa = nn.Conv2d(channels, channels, 3, 1, 1)
        self.fre_att = Attention(dim=channels)

        self.fuse = nn.Sequential(
                                  nn.Conv2d(2*channels, channels, 3, 1, 1),
                                  nn.BatchNorm2d(channels),
                                  nn.ReLU(),
                                  nn.Conv2d(channels, 2*channels, 3, 1, 1),
                                  nn.BatchNorm2d(2*channels),
                                  nn.Sigmoid())

    def forward(self, spa, fre):
        fre = self.fre(fre)
        spa = self.spa(spa)

        fre = self.fre_att(fre, spa) + fre
        spa = self.fre_att(spa, fre) + spa

        fuse = self.fuse(torch.cat((fre, spa), 1))
        fre_a, spa_a = fuse.chunk(2, dim=1)
        spa = spa_a * spa
        fre = fre * fre_a
        res = fre + spa

        return res

    def normalize_feature(self, x):
        return x - x.mean(1).unsqueeze(1)

    def gaussian_normalize(self, x, dim, eps=1e-05):
        x_mean = torch.mean(x, dim=dim, keepdim=True)
        x_var = torch.var(x, dim=dim, keepdim=True)
        x = torch.div(x - x_mean, torch.sqrt(x_var + eps))
        return x


if __name__ == '__main__':
    spa_feat = torch.randn(80, 64, 21, 21)
    freq_feat = torch.randn(80, 64, 21, 21)
    tem = Vit(64, 2, 2, 64, 5)
