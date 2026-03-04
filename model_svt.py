import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from svt import SemanticVideoTransformer


class TwoStreamSVT_CLIP(nn.Module):
    """
    Compatibility wrapper for SemanticVideoTransformer so the training/eval code
    can keep the same output contract as TwoStream* models.

    Input:
      mhi_bcthw:  [B, C_mhi, Tm, Hm, Wm]
      flow_bcthw: [B, C_flow, Tf, Hf, Wf]
    Output dict keys:
      emb_fuse, emb_fuse_raw, logits_cls
    """

    def __init__(
        self,
        *,
        mhi_channels: int,
        flow_channels: int = 2,
        mhi_frames: int = 64,
        flow_frames: int = 64,
        img_size: int = 224,
        embed_dim: int = 768,
        semantic_dim: int = 512,
        patch_size: int = 16,
        depth: int = 12,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        max_frames: Optional[int] = None,
        motion_mask_enabled: bool = False,
        motion_keep_ratio: float = 0.5,
        motion_score_mode: str = "mhi_flow",
        motion_mhi_weight: float = 1.0,
        motion_eps: float = 1e-6,
        num_classes: int = 0,
        compute_second_only: bool = False,
        active_branch: str = "both",
    ) -> None:
        super().__init__()
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

        self.mhi_channels = int(mhi_channels)
        self.flow_channels = int(flow_channels)
        self.img_size = int(img_size)
        self.max_frames = int(max_frames) if max_frames is not None else int(max(mhi_frames, flow_frames))
        self.semantic_dim = int(semantic_dim)
        self.embed_dim = int(embed_dim)

        self.backbone = SemanticVideoTransformer(
            img_size=self.img_size,
            patch_size=int(patch_size),
            in_chans=3,
            max_frames=self.max_frames,
            embed_dim=int(embed_dim),
            depth=int(depth),
            num_heads=int(num_heads),
            mlp_ratio=float(mlp_ratio),
            attn_drop=float(attn_drop),
            proj_drop=float(proj_drop),
            semantic_dim=self.semantic_dim,
            motion_mask_enabled=bool(motion_mask_enabled),
            motion_keep_ratio=float(motion_keep_ratio),
            motion_score_mode=str(motion_score_mode),
            motion_mhi_weight=float(motion_mhi_weight),
            motion_eps=float(motion_eps),
        )

        # Train class-separation directly from the CLS token features.
        self.cls_head = nn.Linear(self.embed_dim, int(num_classes)) if int(num_classes) > 0 else None

    @staticmethod
    def _temporal_resample(x: torch.Tensor, t_out: int) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if t == t_out:
            return x
        idx = torch.linspace(0, t - 1, steps=t_out, device=x.device)
        idx = torch.round(idx).long().clamp_(0, t - 1)
        return x.index_select(2, idx)

    @staticmethod
    def _spatial_resize(x: torch.Tensor, size: int) -> torch.Tensor:
        b, c, t, h, w = x.shape
        if h == size and w == size:
            return x
        x_btchw = x.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w)
        x_btchw = F.interpolate(x_btchw, size=(size, size), mode="bilinear", align_corners=False)
        return x_btchw.reshape(b, t, c, size, size).permute(0, 2, 1, 3, 4).contiguous()

    def _prepare_mhi(self, mhi_bcthw: Optional[torch.Tensor], b: int, t: int, device, dtype) -> torch.Tensor:
        if mhi_bcthw is None or not self.has_top:
            return torch.zeros((b, 1, t, self.img_size, self.img_size), device=device, dtype=dtype)
        mhi = self._temporal_resample(mhi_bcthw, t)
        mhi = self._spatial_resize(mhi, self.img_size)
        if mhi.shape[1] == 1:
            return mhi
        return mhi.mean(dim=1, keepdim=True)

    def _prepare_flow(self, flow_bcthw: Optional[torch.Tensor], b: int, t: int, device, dtype) -> torch.Tensor:
        if flow_bcthw is None or not self.has_bot:
            return torch.zeros((b, 2, t, self.img_size, self.img_size), device=device, dtype=dtype)
        flow = self._temporal_resample(flow_bcthw, t)
        flow = self._spatial_resize(flow, self.img_size)
        c = flow.shape[1]
        if c < 2:
            pad = torch.zeros((b, 2 - c, t, self.img_size, self.img_size), device=device, dtype=flow.dtype)
            flow = torch.cat([flow, pad], dim=1)
        elif c > 2:
            flow = flow[:, :2]
        return flow

    def _to_svt_input(
        self,
        mhi_bcthw: Optional[torch.Tensor],
        flow_bcthw: Optional[torch.Tensor],
    ) -> torch.Tensor:
        if mhi_bcthw is None and flow_bcthw is None:
            raise ValueError("Both mhi and flow inputs are None.")

        ref = mhi_bcthw if mhi_bcthw is not None else flow_bcthw
        b = int(ref.shape[0])
        device = ref.device
        dtype = ref.dtype

        t_candidates = []
        if mhi_bcthw is not None and self.has_top:
            t_candidates.append(int(mhi_bcthw.shape[2]))
        if flow_bcthw is not None and self.has_bot:
            t_candidates.append(int(flow_bcthw.shape[2]))
        if not t_candidates:
            t_candidates.append(int(ref.shape[2]))
        t = min(t_candidates)
        t = max(1, min(t, self.max_frames))

        mhi = self._prepare_mhi(mhi_bcthw, b, t, device, dtype)
        flow = self._prepare_flow(flow_bcthw, b, t, device, dtype)
        x_bcthw = torch.cat([mhi, flow], dim=1)  # [B,3,T,H,W]
        return x_bcthw.permute(0, 2, 1, 3, 4).contiguous()  # [B,T,3,H,W]

    def forward(self, mhi_bcthw: Optional[torch.Tensor], flow_bcthw: Optional[torch.Tensor]):
        x = self._to_svt_input(mhi_bcthw, flow_bcthw)
        cls_token = self.backbone.forward_features(x)      # [B, embed_dim]
        output = self.backbone.semantic_head(cls_token)    # [B, semantic_dim]
        logits_cls = self.cls_head(cls_token) if self.cls_head is not None else None

        return {
            "emb_fuse": output,
            "emb_fuse_raw": output,
            "logits_cls": logits_cls,
        }
