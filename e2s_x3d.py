import torch
import torch.nn as nn

# -----------------------------
# OPTIONAL deps for FLOPs
# -----------------------------
try:
    from fvcore.nn import FlopCountAnalysis
    _HAS_FVCORE = True
except Exception:
    _HAS_FVCORE = False

try:
    from pytorchvideo.models.x3d import create_x3d
    _HAS_PYTORCHVIDEO = True
except Exception:
    _HAS_PYTORCHVIDEO = False


def build_x3d_embedder(
    *,
    in_ch: int,
    clip_len: int,
    crop: int,
    embed_dim: int,
    dropout: float,
    width_factor: float,
    depth_factor: float,
    head_dim_out: int,
) -> nn.Module:
    """
    X3D model that outputs (B, embed_dim) by setting model_num_class=embed_dim
    and head_activation=None (embedding head).
    """
    if not _HAS_PYTORCHVIDEO:
        raise RuntimeError("pytorchvideo not installed. `pip install pytorchvideo`")

    return create_x3d(
        input_channel=in_ch,
        input_clip_length=clip_len,
        input_crop_size=crop,
        model_num_class=embed_dim,                # output embedding
        dropout_rate=dropout,
        width_factor=width_factor,
        depth_factor=depth_factor,
        head_dim_out=head_dim_out,
        head_activation=None,                     # no softmax
        head_output_with_global_average=True,
    )

class TwoStreamE2S_X3D_CLIP(nn.Module):
    """
    Top:  MHI clip  (B, C_mhi, Tm, img_size, img_size)
    Bot:  Flow clip (B, C_flow, Tf, flow_hw, flow_hw)

    Outputs:
      emb_top  (B, embed_dim)
      emb_bot  (B, embed_dim)
      emb_fuse (B, embed_dim)
    """
    def __init__(
        self,
        mhi_channels: int,
        flow_channels: int = 2,
        mhi_frames: int = 32,
        flow_frames: int = 128,
        img_size: int = 224,
        flow_hw: int = 112,
        embed_dim: int = 512,
        fuse: str = "concat",   # "avg_then_proj" or "concat"
        dropout: float = 0.0,
        # X3D knobs (use S-ish defaults; tune as you like)
        top_width_factor: float = 2.0,
        top_depth_factor: float = 2.2,
        bot_width_factor: float = 2.0,
        bot_depth_factor: float = 2.2,
        top_head_dim_out: int = 2048,
        bot_head_dim_out: int = 2048,
        compute_second_only: bool = False,
    ):
        super().__init__()
        self.compute_second_only = compute_second_only
        self.fuse = fuse

        if not compute_second_only:
            self.top = build_x3d_embedder(
                in_ch=mhi_channels,
                clip_len=mhi_frames,
                crop=img_size,
                embed_dim=embed_dim,
                dropout=dropout,
                width_factor=top_width_factor,
                depth_factor=top_depth_factor,
                head_dim_out=top_head_dim_out,
            )
        self.bot = build_x3d_embedder(
            in_ch=flow_channels,
            clip_len=flow_frames,
            crop=flow_hw,
            embed_dim=embed_dim,
            dropout=dropout,
            width_factor=bot_width_factor,
            depth_factor=bot_depth_factor,
            head_dim_out=bot_head_dim_out,
        )

        if fuse == "concat":
            self.proj_fuse = nn.Linear(embed_dim * 2, embed_dim)
        elif fuse == "avg_then_proj":
            self.proj_fuse = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError("fuse must be one of: concat, avg_then_proj")

    def forward(self, mhi_bcthw: torch.Tensor, flow_bcthw: torch.Tensor):
        eb = self.bot(flow_bcthw)
        if (not self.compute_second_only) and (mhi_bcthw is not None):
            et = self.top(mhi_bcthw)
        else:
            et = torch.zeros_like(eb)

        if self.fuse == "concat":
            ef = self.proj_fuse(torch.cat([et, eb], dim=-1))
        else:
            ef = self.proj_fuse(0.5 * (et + eb))
        return {"emb_top": et, "emb_bot": eb, "emb_fuse": ef}