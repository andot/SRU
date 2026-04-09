import torch.nn as nn


class SRVGGNetCompact(nn.Module):
    def __init__(
        self,
        num_in_ch=3,
        num_out_ch=3,
        num_feat=64,
        num_conv=16,
        upscale=4,
        act_type="prelu",
    ):
        super().__init__()
        self.num_in_ch = num_in_ch
        self.num_out_ch = num_out_ch
        self.num_feat = num_feat
        self.num_conv = num_conv
        self.upscale = upscale
        self.act_type = act_type

        mods = nn.ModuleList()
        mods.append(nn.Conv2d(num_in_ch, num_feat, 3, 1, 1))
        mods.append(SRVGGNetCompact.get_activation(num_feat, act_type))

        for _ in range(num_conv):
            mods.append(nn.Conv2d(num_feat, num_feat, 3, 1, 1))
            mods.append(SRVGGNetCompact.get_activation(num_feat, act_type))

        mods.append(nn.Conv2d(num_feat, num_out_ch * upscale * upscale, 3, 1, 1))
        mods.append(nn.PixelShuffle(upscale))

        self.body = nn.Sequential(*mods)

        # Use nn.Upsample (Module) instead of F.interpolate for better CoreML compatibility
        self.base_upsampler = nn.Upsample(
            scale_factor=float(upscale), mode="nearest", recompute_scale_factor=False
        )

    @staticmethod
    def get_activation(num_feat, act_type):
        match act_type:
            case "relu":
                return nn.ReLU(inplace=True)
            case "prelu":
                return nn.PReLU(num_parameters=num_feat)
            case "leakyrelu":
                return nn.LeakyReLU(negative_slope=0.1, inplace=True)
            case _:
                raise ValueError(f"Unsupported activation type: {act_type}")

    def forward(self, x):
        return self.body(x) + self.base_upsampler(x)


__all__ = ["SRVGGNetCompact"]
