"""Local model architecture package (RRDB / SRVGG)"""

import torch


def build_net(model_name: str) -> torch.nn.Module:
    mn = model_name.split(".")[0]
    net = None
    import_locations = []

    # Use local archs implementations (archs/rrdb.py, archs/srvgg.py).
    # Fail loudly if these cannot be imported so issues are visible.
    from .srvgg import SRVGGNetCompact
    from .rrdb import RRDBNet

    import_locations.append("archs.srvgg.SRVGGNetCompact")
    import_locations.append("archs.rrdb.RRDBNet")

    # Select appropriate constructor
    if mn == "RealESRGAN_x4plus_anime_6B" and RRDBNet is not None:
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=6,
            num_grow_ch=32,
        )
    elif mn == "RealESRGAN_x4plus" and RRDBNet is not None:
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )
    elif mn == "RealESRNet_x4plus" and RRDBNet is not None:
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=4,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )
    elif mn == "RealESRGAN_x2plus" and RRDBNet is not None:
        net = RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            scale=2,
            num_feat=64,
            num_block=23,
            num_grow_ch=32,
        )
    elif mn == "realesr-animevideov3" and SRVGGNetCompact is not None:
        net = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
    elif (
        mn in ("realesr-general-x4v3", "realesr-general-wdn-x4v3")
        and SRVGGNetCompact is not None
    ):
        net = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
    elif mn == "RealESRGANv2-animevideo-xsx4" and SRVGGNetCompact is not None:
        net = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=4,
            act_type="prelu",
        )
    elif mn == "RealESRGANv2-animevideo-xsx2" and SRVGGNetCompact is not None:
        net = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=16,
            upscale=2,
            act_type="prelu",
        )

    if net is None:
        msg = f"Could not build network for model name: {model_name}. "
        msg += "Imported modules found: " + ",".join(import_locations)
        raise RuntimeError(msg)
    return net


__all__ = ["rrdb", "srvgg", "build_net"]
