import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualDenseBlock_5C(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock_5C, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, 1, 1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, 1, 1)
        self.conv3 = nn.Conv2d(nf + gc * 2, gc, 3, 1, 1)
        self.conv4 = nn.Conv2d(nf + gc * 3, gc, 3, 1, 1)
        self.conv5 = nn.Conv2d(nf + gc * 4, nf, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    def __init__(self, nf, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock_5C(nf, gc)
        self.rdb2 = ResidualDenseBlock_5C(nf, gc)
        self.rdb3 = ResidualDenseBlock_5C(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


class RRDBNet(nn.Module):
    """CoreML-safe RRDBNet matching basicsr parameter names exactly.
    Uses nn.Upsample instead of F.interpolate, and manual reshape+permute
    for pixel unshuffle when needed.
    """

    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        scale=4,
        num_feat=64,
        num_block=23,
        num_grow_ch=32,
    ):
        super(RRDBNet, self).__init__()
        self.scale = scale
        # upsample layers — use nn.Upsample (Module) instead of F.interpolate
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        # Match basicsr: pixel unshuffle expands channels for scale < 4
        self._unshuffle_factor = 0
        if scale == 2:
            self._unshuffle_factor = 2
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            self._unshuffle_factor = 4
            num_in_ch = num_in_ch * 16
        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat, num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        # always create both conv_up layers (matching basicsr)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=False)

    def forward(self, x):
        if self._unshuffle_factor > 0:
            x = F.pixel_unshuffle(x, self._unshuffle_factor)
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        feat = self.lrelu(self.conv_up1(self.upsample(feat)))
        feat = self.lrelu(self.conv_up2(self.upsample(feat)))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


__all__ = ["RRDBNet", "RRDB", "ResidualDenseBlock_5C"]
