import torch
import torch.nn as nn
from typing import Optional

try:
    from pytorchvideo.models.x3d import create_x3d
    _HAS_PYTORCHVIDEO = True
except Exception:
    _HAS_PYTORCHVIDEO = False

class MLPProjector(nn.Module):
    def __init__(self, in_dim=2048, hidden_dim=2048, out_dim=512, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
            nn.LayerNorm(out_dim),
        )

    def forward(self, x):
        return self.net(x)
    

def build_x3d_embedder(
    *,
    in_ch: int,
    clip_len: int,
    crop: int,
    output_dim: int,
    dropout: float,
    width_factor: float,
    depth_factor: float,
    head_dim_out: int,
) -> nn.Module:
    """
    X3D model that outputs (B, output_dim) by setting model_num_class=output_dim
    and head_activation=None (embedding head).
    """
    if not _HAS_PYTORCHVIDEO:
        raise RuntimeError("pytorchvideo not installed. `pip install pytorchvideo`")

    return create_x3d(
        input_channel=in_ch,
        input_clip_length=clip_len,
        input_crop_size=crop,
        model_num_class=output_dim,                # output embedding
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
        use_projection: bool = False,
        dual_projection_heads: bool = False,
        num_classes: int = 0,
        projection_dropout: float = 0.5,
        use_nonlinear_projection: Optional[bool] = None,  # legacy alias
        active_branch: str = "both",
    ):
        super().__init__()
        if use_nonlinear_projection is not None:
            use_projection = bool(use_projection or use_nonlinear_projection)
        if compute_second_only:
            if active_branch not in ("both", "second"):
                raise ValueError("Conflicting branch settings: compute_second_only=True and active_branch!='second'")
            active_branch = "second"
        if active_branch not in ("both", "first", "second"):
            raise ValueError(f"active_branch must be one of: both, first, second (got: {active_branch})")

        self.active_branch = active_branch
        self.has_top = active_branch in ("both", "first")
        self.has_bot = active_branch in ("both", "second")
        self.compute_second_only = (active_branch == "second")
        self.fuse = fuse
        self.use_projection = bool(use_projection)
        self.dual_projection_heads = bool(self.use_projection and dual_projection_heads)

        self.top = None
        self.bot = None
        self.proj_top = None
        self.proj_bot = None

        if self.has_top:
            self.top = build_x3d_embedder(
                in_ch=mhi_channels,
                clip_len=mhi_frames,
                crop=img_size,
                output_dim=embed_dim*2,
                dropout=dropout,
                width_factor=top_width_factor,
                depth_factor=top_depth_factor,
                head_dim_out=top_head_dim_out,
            )
        if self.has_bot:
            self.bot = build_x3d_embedder(
                in_ch=flow_channels,
                clip_len=flow_frames,
                crop=flow_hw,
                output_dim=embed_dim*2,
                dropout=dropout,
                width_factor=bot_width_factor,
                depth_factor=bot_depth_factor,
                head_dim_out=bot_head_dim_out,
            )

        if self.has_top:
            self.proj_top = nn.Linear(embed_dim*2, embed_dim)
        if self.has_bot:
            self.proj_bot = nn.Linear(embed_dim*2, embed_dim)

        if fuse == "concat":
            self.proj_fuse = nn.Linear(embed_dim*2, embed_dim)
        elif fuse == "avg_then_proj":
            self.proj_fuse = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError("fuse must be one of: concat, avg_then_proj")

        self.clip_head = None
        self.embed_head = None
        self.cls_head = None
        if self.use_projection:
            self.clip_head = nn.Sequential(
                nn.LayerNorm(embed_dim),
                nn.Linear(embed_dim, embed_dim),
            )
            if self.dual_projection_heads:
                self.embed_head = nn.Sequential(
                    nn.LayerNorm(embed_dim),
                    nn.Linear(embed_dim, embed_dim),
                )
            if int(num_classes) > 0:
                self.cls_head = nn.Sequential(
                    nn.Dropout(float(projection_dropout)),
                    nn.Linear(embed_dim, int(num_classes)),
                )

    def forward(self, mhi_bcthw: torch.Tensor, flow_bcthw: torch.Tensor):
        et = None
        eb = None

        if self.has_top:
            if mhi_bcthw is None:
                raise ValueError("active_branch requires first/top branch, but mhi input is None")
            et = self.proj_top(self.top(mhi_bcthw))

        if self.has_bot:
            if flow_bcthw is None:
                raise ValueError("active_branch requires second/bot branch, but flow input is None")
            eb = self.proj_bot(self.bot(flow_bcthw))

        if et is None and eb is None:
            raise RuntimeError("Model has no active branches. Set active_branch to both, first, or second.")
        if et is None:
            et = torch.zeros_like(eb)
        if eb is None:
            eb = torch.zeros_like(et)

        if self.fuse == "concat":
            ef_raw = self.proj_fuse(torch.cat([et, eb], dim=-1))
        else:
            ef_raw = self.proj_fuse(0.5 * (et + eb))
        ef_clip = self.clip_head(ef_raw) if self.clip_head is not None else ef_raw
        ef_embed = self.embed_head(ef_raw) if self.embed_head is not None else ef_clip
        logits_cls = self.cls_head(ef_raw) if self.cls_head is not None else None
        return {
            "emb_top": et,
            "emb_bot": eb,
            "emb_fuse": ef_clip,
            "emb_fuse_clip": ef_clip,
            "emb_fuse_embed": ef_embed,
            "emb_fuse_raw": ef_raw,
            "logits_cls": logits_cls,
        }
