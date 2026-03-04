import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Iterable, Tuple


def drop_path(x: torch.Tensor, drop_prob: float = 0.0, training: bool = False) -> torch.Tensor:
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0) -> None:
        super().__init__()
        self.drop_prob = float(drop_prob)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return drop_path(x, self.drop_prob, self.training)


class FeedForward(nn.Module):
    """Standard Transformer MLP block."""
    def __init__(
        self,
        dim: int,
        hidden_dim: int,
        out_dim: int | None = None,
        drop: float = 0.0,
        act_layer: type[nn.Module] = nn.GELU,
    ) -> None:
        super().__init__()
        out_dim = out_dim or dim
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SemanticHead(nn.Module):
    """
    Paper-faithful semantic head:
    final summary token -> MLP with 3 hidden layers -> 600-d semantic embedding.
    Hidden widths are configurable because the paper text does not pin them down
    in the lines we can verify.
    """
    def __init__(
        self,
        in_dim: int,
        out_dim: int = 600,
        hidden_dims: Iterable[int] = (2048, 2048, 1024),
    ) -> None:
        super().__init__()
        dims = [in_dim, *hidden_dims, out_dim]
        layers = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class VideoPatchEmbed(nn.Module):
    """
    Patchify each frame with a Conv2d whose kernel=stride=patch_size.
    This is equivalent to flatten+linear-project per patch.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 768,
    ) -> None:
        super().__init__()
        if img_size % patch_size != 0:
            raise ValueError("img_size must be divisible by patch_size.")

        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        returns: [B, T, N, D]
        """
        if x.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got shape {tuple(x.shape)}")

        b, t, c, h, w = x.shape
        if c != self.in_chans:
            raise ValueError(f"Expected {self.in_chans} channels, got {c}")
        if h != self.img_size or w != self.img_size:
            raise ValueError(
                f"Expected spatial size {self.img_size}x{self.img_size}, got {h}x{w}"
            )

        x = x.reshape(b * t, c, h, w)                # [B*T, C, H, W]
        x = self.proj(x)                             # [B*T, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)             # [B*T, N, D]
        x = x.reshape(b, t, self.num_patches, self.embed_dim)
        return x


class DividedSpaceTimeBlock(nn.Module):
    """
    TimeSformer-style divided attention:
      1) temporal attention across frames at each spatial location
      2) spatial attention across patches in each frame (+ cls token)
      3) token MLP
    """
    def __init__(
        self,
        dim: int = 768,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_prob: float = 0.0,
    ) -> None:
        super().__init__()
        mlp_hidden_dim = int(dim * mlp_ratio)

        self.norm_t = nn.LayerNorm(dim)
        self.temporal_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )
        # TimeSformer applies a projection after temporal attention
        # before adding it back to the residual stream.
        self.temporal_fc = nn.Linear(dim, dim)

        self.norm_s = nn.LayerNorm(dim)
        self.spatial_attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=attn_drop,
            batch_first=True,
        )

        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = FeedForward(
            dim=dim,
            hidden_dim=mlp_hidden_dim,
            out_dim=dim,
            drop=proj_drop,
            act_layer=nn.GELU,
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0.0 else nn.Identity()

    def forward(
        self,
        patch_tokens: torch.Tensor,  # [B, T, N, D]
        cls_token: torch.Tensor,     # [B, 1, D]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        b, t, n, d = patch_tokens.shape

        # ---- 1) Temporal attention (per spatial location across time) ----
        xt = self.norm_t(patch_tokens)               # [B, T, N, D]
        xt = xt.permute(0, 2, 1, 3).reshape(b * n, t, d)  # [B*N, T, D]
        xt, _ = self.temporal_attn(xt, xt, xt, need_weights=False)
        xt = self.temporal_fc(xt)
        xt = xt.reshape(b, n, t, d).permute(0, 2, 1, 3)   # [B, T, N, D]
        patch_tokens = patch_tokens + self.drop_path(xt)

        # ---- 2) Spatial attention (per frame across patches + cls) ----
        cls_rep = cls_token.unsqueeze(1).expand(-1, t, -1, -1)  # [B, T, 1, D]
        xs = torch.cat([cls_rep, patch_tokens], dim=2)          # [B, T, N+1, D]
        xs = self.norm_s(xs).reshape(b * t, n + 1, d)           # [B*T, N+1, D]
        xs, _ = self.spatial_attn(xs, xs, xs, need_weights=False)
        xs = xs.reshape(b, t, n + 1, d)                         # [B, T, N+1, D]

        cls_update = xs[:, :, 0, :].mean(dim=1, keepdim=True)   # [B, 1, D]
        patch_update = xs[:, :, 1:, :]                          # [B, T, N, D]

        cls_token = cls_token + self.drop_path(cls_update)
        patch_tokens = patch_tokens + self.drop_path(patch_update)

        # ---- 3) MLP on all tokens ----
        all_tokens = torch.cat(
            [cls_token, patch_tokens.reshape(b, t * n, d)], dim=1
        )  # [B, 1+T*N, D]
        all_tokens = all_tokens + self.drop_path(self.mlp(self.norm_mlp(all_tokens)))

        cls_token = all_tokens[:, :1, :]
        patch_tokens = all_tokens[:, 1:, :].reshape(b, t, n, d)
        return patch_tokens, cls_token


class SemanticVideoTransformer(nn.Module):
    """
    Paper-faithful SVT backbone + semantic head.

    Input:
        [B, T, 3, 224, 224]  where T <= max_frames (e.g. 96)
    Output:
        [B, 600] semantic embedding

    Notes:
    - `forward_features()` returns the final summary token before the semantic head.
    - `in_chans=3` means you can feed [MHI, flow_x, flow_y] per timestep directly.
    """
    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        max_frames: int = 96,
        embed_dim: int = 768,
        depth: int = 12,              # configurable
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        drop_path_rate: float = 0.1,
        semantic_dim: int = 600,
        semantic_hidden_dims: Iterable[int] = (2048, 2048, 1024),
        motion_mask_enabled: bool = False,
        motion_keep_ratio: float = 0.5,
        motion_score_mode: str = "mhi_flow",
        motion_mhi_weight: float = 1.0,
        motion_eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.max_frames = max_frames
        self.embed_dim = embed_dim
        self.motion_mask_enabled = bool(motion_mask_enabled)
        self.motion_keep_ratio = float(motion_keep_ratio)
        self.motion_score_mode = str(motion_score_mode)
        self.motion_mhi_weight = float(motion_mhi_weight)
        self.motion_eps = float(motion_eps)
        if not (0.0 < self.motion_keep_ratio <= 1.0):
            raise ValueError("motion_keep_ratio must be in (0, 1].")
        if self.motion_score_mode not in ("mhi_flow", "l1_mean"):
            raise ValueError("motion_score_mode must be one of: mhi_flow, l1_mean")

        self.patch_embed = VideoPatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        self.num_patches = self.patch_embed.num_patches

        # Learned summary token z_0^0(0)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        # TimeSformer-style positional embeddings:
        # one spatial embedding table (cls + patches) and one temporal table.
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, embed_dim))
        self.time_embed = nn.Parameter(torch.zeros(1, max_frames, embed_dim))
        self.pos_drop = nn.Dropout(p=proj_drop)

        dpr = torch.linspace(0, float(drop_path_rate), steps=depth).tolist() if depth > 0 else []
        self.blocks = nn.ModuleList(
            [
                DividedSpaceTimeBlock(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    attn_drop=attn_drop,
                    proj_drop=proj_drop,
                    drop_path_prob=float(dpr[i]),
                )
                for i in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim)
        self.semantic_head = SemanticHead(
            in_dim=embed_dim,
            out_dim=semantic_dim,
            hidden_dims=tuple(semantic_hidden_dims),
        )

        self._init_weights()
        self._init_temporal_attention_residual()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.time_embed, std=0.02)

        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="linear")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def _init_temporal_attention_residual(self) -> None:
        # Match TimeSformer initialization behavior:
        # keep first block temporal_fc learned, zero-init deeper block temporal_fc
        # so the model starts closer to an image-ViT prior.
        for i, blk in enumerate(self.blocks):
            if i == 0:
                continue
            if hasattr(blk, "temporal_fc"):
                nn.init.constant_(blk.temporal_fc.weight, 0.0)
                if blk.temporal_fc.bias is not None:
                    nn.init.constant_(blk.temporal_fc.bias, 0.0)

    def _compute_motion_score_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W] -> score map: [B, T, H, W]
        """
        if self.motion_score_mode == "l1_mean" or x.shape[2] < 3:
            return x.abs().mean(dim=2)

        mhi = x[:, :, 0, :, :].abs()
        flow_x = x[:, :, 1, :, :]
        flow_y = x[:, :, 2, :, :]
        flow_mag = torch.sqrt(flow_x * flow_x + flow_y * flow_y + self.motion_eps)
        return self.motion_mhi_weight * mhi + flow_mag

    def _compute_patch_scores(self, score_map: torch.Tensor) -> torch.Tensor:
        """
        score_map: [B, T, H, W] -> patch scores: [B, T, N]
        """
        b, t, h, w = score_map.shape
        score_bt = score_map.reshape(b * t, 1, h, w)
        pooled = F.avg_pool2d(
            score_bt,
            kernel_size=self.patch_embed.patch_size,
            stride=self.patch_embed.patch_size,
        )
        return pooled.flatten(1).reshape(b, t, self.num_patches)

    def _select_topk_indices(self, patch_scores: torch.Tensor) -> Tuple[torch.Tensor, int]:
        """
        patch_scores: [B, T, N] -> selected indices [B, T, k]
        """
        # Use model metadata to keep `k` as a Python int even under JIT tracing
        # (fvcore/torch.jit can turn shape-derived values into Tensor-like objects).
        n = self.num_patches
        k = max(1, int(round(self.motion_keep_ratio * n)))
        k = min(k, n)
        idx = torch.topk(patch_scores, k=k, dim=-1, largest=True, sorted=False).indices
        idx = idx.sort(dim=-1).values
        return idx, k

    def _apply_motion_mask(
        self,
        patch_tokens: torch.Tensor,   # [B, T, N, D]
        idx: torch.Tensor,            # [B, T, k]
    ) -> torch.Tensor:
        """
        Returns:
            masked patch tokens flattened [B, T*k, D]
        """
        b, t, _, d = patch_tokens.shape
        k = int(idx.shape[-1])

        idx_exp = idx.unsqueeze(-1).expand(-1, -1, -1, d)
        masked_patch = torch.gather(patch_tokens, dim=2, index=idx_exp).reshape(b, t * k, d)
        return masked_patch

    def forward_features(
        self, x: torch.Tensor, return_mask_info: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, dict]:
        """
        Returns final summary token z_0^0(L): [B, D]
        """
        if x.ndim != 5:
            raise ValueError(f"Expected [B, T, C, H, W], got shape {tuple(x.shape)}")

        b, t, _, _, _ = x.shape
        if t > self.max_frames:
            raise ValueError(f"Input has {t} frames, but max_frames={self.max_frames}")

        patch_tokens = self.patch_embed(x)                      # [B, T, N, D]
        selected_idx = None
        k = self.num_patches

        # Spatial + temporal positional encoding (TimeSformer style).
        cls = self.cls_token.expand(b, -1, -1) + self.pos_embed[:, :1, :]
        patch_tokens = patch_tokens + self.pos_embed[:, 1:, :].unsqueeze(1)
        patch_tokens = patch_tokens + self.time_embed[:, :t, :].unsqueeze(2)
        patch_tokens = self.pos_drop(patch_tokens)

        if self.motion_mask_enabled:
            score_map = self._compute_motion_score_map(x)
            patch_scores = self._compute_patch_scores(score_map)
            selected_idx, k = self._select_topk_indices(patch_scores)
            flat_tokens = self._apply_motion_mask(patch_tokens, selected_idx)
        else:
            flat_tokens = patch_tokens.reshape(b, t * self.num_patches, self.embed_dim)

        tokens = torch.cat([cls, flat_tokens], dim=1)          # [B, 1+T*K, D]

        cls = tokens[:, :1, :]
        patch_tokens = tokens[:, 1:, :].reshape(b, t, k, self.embed_dim)

        for block in self.blocks:
            patch_tokens, cls = block(patch_tokens, cls)

        cls = self.norm(cls)[:, 0, :]                          # [B, D]
        if not return_mask_info:
            return cls
        mask_info = {
            "keep_ratio": self.motion_keep_ratio if self.motion_mask_enabled else 1.0,
            "k_per_frame": k,
            "selected_idx": selected_idx,
        }
        return cls, mask_info

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns 600-d semantic embedding f(x): [B, 600]
        """
        cls = self.forward_features(x)
        semantic = self.semantic_head(cls)
        return semantic


def _format_flops(flops: int) -> str:
    if flops >= 1_000_000_000_000:
        return f"{flops / 1_000_000_000_000:.3f} TFLOPs"
    if flops >= 1_000_000_000:
        return f"{flops / 1_000_000_000:.3f} GFLOPs"
    if flops >= 1_000_000:
        return f"{flops / 1_000_000:.3f} MFLOPs"
    return f"{flops} FLOPs"


def profile_flops(model: nn.Module, x: torch.Tensor) -> int | None:
    """
    Runtime FLOPs from torch.profiler (if available for the ops/build).
    """
    try:
        from torch.profiler import ProfilerActivity, profile
    except Exception:
        return None

    activities = [ProfilerActivity.CPU]
    if x.is_cuda:
        activities.append(ProfilerActivity.CUDA)

    model.eval()
    try:
        with torch.no_grad():
            with profile(activities=activities, with_flops=True, record_shapes=False) as prof:
                _ = model(x)
        total_flops = sum(
            int(getattr(op, "flops", 0) or 0) for op in prof.key_averages()
        )
        return total_flops if total_flops > 0 else None
    except Exception:
        return None


def estimate_flops(
    model: SemanticVideoTransformer,
    num_frames: int,
    batch_size: int = 1,
    keep_ratio_override: float | None = None,
) -> int:
    """
    Analytical FLOPs estimate for one forward pass (inference).
    """
    pe = model.patch_embed
    t = int(num_frames)
    b = int(batch_size)
    n = pe.num_patches
    d = model.embed_dim
    c = pe.in_chans
    p2 = pe.patch_size * pe.patch_size
    if len(model.blocks) == 0:
        raise ValueError("FLOPs estimation requires at least one transformer block.")
    first_block = model.blocks[0]
    h = first_block.temporal_attn.num_heads
    mlp_hidden = first_block.mlp.fc1.out_features

    if keep_ratio_override is None:
        keep_ratio = model.motion_keep_ratio if model.motion_mask_enabled else 1.0
    else:
        keep_ratio = float(keep_ratio_override)
    if not (0.0 < keep_ratio <= 1.0):
        raise ValueError("keep_ratio must be in (0, 1].")
    k = max(1, int(round(keep_ratio * n)))

    l_tokens = 1 + t * k
    flops = 0

    # Patch embedding conv2d on each frame.
    flops += 2 * b * t * n * d * c * p2

    for _ in model.blocks:
        # Temporal attention over T for each selected spatial location.
        flops += 2 * b * k * t * d * (3 * d)  # qkv projections
        flops += 4 * b * k * h * t * t * (d // h)  # qk^T and attn*v
        flops += 2 * b * k * t * d * d  # output projection
        flops += 2 * b * k * t * d * d  # temporal_fc

        # Spatial attention over K+1 for each frame.
        s = k + 1
        flops += 2 * b * t * s * d * (3 * d)  # qkv projections
        flops += 4 * b * t * h * s * s * (d // h)  # qk^T and attn*v
        flops += 2 * b * t * s * d * d  # output projection

        # Transformer MLP on all tokens.
        flops += 2 * b * l_tokens * d * mlp_hidden  # fc1
        flops += 2 * b * l_tokens * mlp_hidden * d  # fc2

    # Semantic head MLP.
    dims = [d]
    for layer in model.semantic_head.net:
        if isinstance(layer, nn.Linear):
            dims.append(layer.out_features)
    for in_dim, out_dim in zip(dims[:-1], dims[1:]):
        flops += 2 * b * in_dim * out_dim

    return int(flops)


if __name__ == "__main__":
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    base_model = SemanticVideoTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        max_frames=96,
        embed_dim=768,
        depth=4,
        num_heads=12,
        semantic_dim=600,
        motion_mask_enabled=False,
    ).to(device).eval()

    masked_model = SemanticVideoTransformer(
        img_size=224,
        patch_size=16,
        in_chans=3,
        max_frames=96,
        embed_dim=768,
        depth=4,
        num_heads=12,
        semantic_dim=600,
        motion_mask_enabled=True,
        motion_keep_ratio=0.5,
        motion_score_mode="mhi_flow",
    ).to(device)
    masked_model.load_state_dict(base_model.state_dict(), strict=False)
    masked_model.eval()

    x = torch.randn(2, 8, 3, 224, 224, device=device)

    with torch.no_grad():
        y_base = base_model(x)
        z_base, info_base = base_model.forward_features(x, return_mask_info=True)
        y_mask = masked_model(x)
        z_mask, info_mask = masked_model.forward_features(x, return_mask_info=True)
    print("semantic embedding (no mask):", y_base.shape)
    print("semantic embedding (mask):", y_mask.shape)
    print("summary token (no mask):", z_base.shape)
    print("summary token (mask):", z_mask.shape)
    print(f"tokens/frame (no mask): {info_base['k_per_frame']}")
    print(f"tokens/frame (mask, keep=0.5): {info_mask['k_per_frame']}")

    # Smoke tests
    assert y_base.shape == y_mask.shape == (2, 600)
    assert info_mask["k_per_frame"] == max(1, int(round(0.5 * base_model.num_patches)))

    x_focus = torch.zeros(1, 2, 3, 224, 224, device=device)
    p = base_model.patch_embed.patch_size
    x_focus[:, :, :, :p, :p] = 10.0
    with torch.no_grad():
        _, focus_info = masked_model.forward_features(x_focus, return_mask_info=True)
    assert focus_info["selected_idx"] is not None
    assert int(focus_info["selected_idx"][0, 0, 0].item()) == 0

    fallback_model = SemanticVideoTransformer(
        img_size=224,
        patch_size=16,
        in_chans=2,
        max_frames=8,
        embed_dim=192,
        depth=1,
        num_heads=3,
        semantic_dim=600,
        motion_mask_enabled=True,
        motion_keep_ratio=0.5,
        motion_score_mode="mhi_flow",  # should fallback to l1_mean when C < 3
    ).to(device).eval()
    with torch.no_grad():
        _ = fallback_model(torch.randn(1, 2, 2, 224, 224, device=device))

    with torch.no_grad():
        _, m1 = masked_model.forward_features(x, return_mask_info=True)
        _, m2 = masked_model.forward_features(x, return_mask_info=True)
    assert torch.equal(m1["selected_idx"], m2["selected_idx"])

    # FLOP analysis (forward only)
    x_prof = torch.randn(16, 64, 3, 224, 224, device=device, dtype=torch.float32)
    base_prof = profile_flops(base_model, x_prof)
    mask_prof = profile_flops(masked_model, x_prof)
    if base_prof is not None:
        print(f"profiled FLOPs (no mask, T=8, B=1): {_format_flops(base_prof)}")
    if mask_prof is not None:
        print(f"profiled FLOPs (mask, T=8, B=1): {_format_flops(mask_prof)}")

    # est_base = estimate_flops(base_model, num_frames=8, batch_size=1)
    # est_mask = estimate_flops(masked_model, num_frames=8, batch_size=1)
    try:
        from fvcore.nn import FlopCountAnalysis

        # flops_base = FlopCountAnalysis(base_model, x_prof).total()
        # print("\n=== FLOP Analysis (forward only) ===")
        # print(f"Total FLOPs: {flops_base / 1e9:.3f} GFLOPs\n")

        flops_mask = FlopCountAnalysis(masked_model, x_prof).total()
        device = torch.device('cuda:0')
        free, total = torch.cuda.mem_get_info(device)
        mem_used_MB = (total - free) / 1024 ** 2
        print(f"mem", mem_used_MB)
        print("\n=== FLOP Analysis (forward only) ===")
        print(f"Total FLOPs: {flops_mask / 1e9:.3f} GFLOPs\n")

    except ImportError:
        print("\n[INFO] fvcore not installed — skipping FLOP analysis.\n")
