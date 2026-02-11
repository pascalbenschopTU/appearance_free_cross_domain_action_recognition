import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPProjector(nn.Module):
    def __init__(self, in_dim=1024, hidden_dim=2048, out_dim=512, dropout=0.1):
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

# ----------------------------
# I3D code
# ----------------------------

def get_padding_shape(filter_shape, stride):
    def _pad_top_bottom(filter_dim, stride_val):
        pad_along = max(filter_dim - stride_val, 0)
        pad_top = pad_along // 2
        pad_bottom = pad_along - pad_top
        return pad_top, pad_bottom

    padding_shape = []
    for filter_dim, stride_val in zip(filter_shape, stride):
        pad_top, pad_bottom = _pad_top_bottom(filter_dim, stride_val)
        padding_shape.append(pad_top)
        padding_shape.append(pad_bottom)
    depth_top = padding_shape.pop(0)
    depth_bottom = padding_shape.pop(0)
    padding_shape.append(depth_top)
    padding_shape.append(depth_bottom)

    return tuple(padding_shape)


def simplify_padding(padding_shapes):
    all_same = True
    padding_init = padding_shapes[0]
    for pad in padding_shapes[1:]:
        if pad != padding_init:
            all_same = False
    return all_same, padding_init


class Unit3Dpy(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=(1, 1, 1),
        stride=(1, 1, 1),
        activation="relu",
        padding="SAME",
        use_bias=False,
        use_bn=True,
    ):
        super().__init__()

        self.padding = padding
        self.use_bn = use_bn

        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            simplify_pad, pad_size = simplify_padding(padding_shape)
            self.simplify_pad = simplify_pad
        elif padding == "VALID":
            padding_shape = 0
        else:
            raise ValueError(f"padding should be in [VALID|SAME] but got {padding}")

        if padding == "SAME":
            if not simplify_pad:
                self.pad = nn.ConstantPad3d(padding_shape, 0)
                self.conv3d = nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride=stride, bias=use_bias
                )
            else:
                self.conv3d = nn.Conv3d(
                    in_channels, out_channels, kernel_size, stride=stride, padding=pad_size, bias=use_bias
                )
        elif padding == "VALID":
            self.conv3d = nn.Conv3d(
                in_channels, out_channels, kernel_size, padding=padding_shape, stride=stride, bias=use_bias
            )

        if self.use_bn:
            self.batch3d = nn.BatchNorm3d(out_channels, eps=1e-3, momentum=0.01)

        self.activation = activation

    def forward(self, inp):
        if self.padding == "SAME" and self.simplify_pad is False:
            inp = self.pad(inp)
        out = self.conv3d(inp)
        if self.use_bn:
            out = self.batch3d(out)
        if self.activation is not None:
            out = F.relu(out, inplace=True)
        return out


class InputStem3D(nn.Module):
    """
    Input normalization + first conv. Optionally keeps TF-SAME padding through Unit3Dpy.
    """
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 64,
        *,
        norm: str = "bn",                 # "bn" | "in" | "gn" | "none"
        kernel_size=(7, 7, 7),
        stride=(2, 2, 2),
        padding="SAME",
        use_bias=False,
        use_bn=True,
        activation="relu",
    ):
        super().__init__()

        # input norm (before conv)
        if norm == "bn":
            self.in_norm = nn.BatchNorm3d(in_channels, eps=1e-3, momentum=0.01)
        elif norm == "in":
            self.in_norm = nn.InstanceNorm3d(in_channels, affine=True)
        elif norm == "gn":
            # group=1 works fine for low channel counts too
            self.in_norm = nn.GroupNorm(num_groups=1, num_channels=in_channels)
        elif norm == "none" or norm is None:
            self.in_norm = None
        else:
            raise ValueError(f"Unknown norm: {norm}")

        # first conv block
        self.conv = Unit3Dpy(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            use_bias=use_bias,
            use_bn=use_bn,
            activation=activation,
        )

    def forward(self, x):
        if self.in_norm is not None:
            x = self.in_norm(x)
        return self.conv(x)



class MaxPool3dTFPadding(nn.Module):
    def __init__(self, kernel_size, stride=None, padding="SAME"):
        super().__init__()
        if padding == "SAME":
            padding_shape = get_padding_shape(kernel_size, stride)
            self.pad = nn.ConstantPad3d(padding_shape, 0)
        else:
            self.pad = None
        self.pool = nn.MaxPool3d(kernel_size, stride, ceil_mode=True)

    def forward(self, inp):
        if self.pad is not None:
            inp = self.pad(inp)
        return self.pool(inp)


class Mixed(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch_0 = Unit3Dpy(in_channels, out_channels[0], kernel_size=(1, 1, 1))

        branch_1_conv1 = Unit3Dpy(in_channels, out_channels[1], kernel_size=(1, 1, 1))
        branch_1_conv2 = Unit3Dpy(out_channels[1], out_channels[2], kernel_size=(3, 3, 3))
        self.branch_1 = nn.Sequential(branch_1_conv1, branch_1_conv2)

        branch_2_conv1 = Unit3Dpy(in_channels, out_channels[3], kernel_size=(1, 1, 1))
        branch_2_conv2 = Unit3Dpy(out_channels[3], out_channels[4], kernel_size=(3, 3, 3))
        self.branch_2 = nn.Sequential(branch_2_conv1, branch_2_conv2)

        branch_3_pool = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(1, 1, 1), padding="SAME")
        branch_3_conv2 = Unit3Dpy(in_channels, out_channels[5], kernel_size=(1, 1, 1))
        self.branch_3 = nn.Sequential(branch_3_pool, branch_3_conv2)

    def forward(self, inp):
        out_0 = self.branch_0(inp)
        out_1 = self.branch_1(inp)
        out_2 = self.branch_2(inp)
        out_3 = self.branch_3(inp)
        return torch.cat((out_0, out_1, out_2, out_3), 1)


class I3DFeature(nn.Module):
    """
    I3D trunk -> global pooled feature (B,1024)
    Uses the same topology as your reference I3D, but with AdaptiveAvgPool3d(1)
    so it works cleanly for both:
      - (T=32, H=W=224)
      - (T=128, H=W=112)
    """
    def __init__(self, in_channels: int, dropout_prob: float = 0.0, stem=None):
        super().__init__()
        self.stem = stem

        self.conv3d_1a_7x7 = Unit3Dpy(
            out_channels=64,
            in_channels=in_channels,
            kernel_size=(7, 7, 7),
            stride=(2, 2, 2),
            padding="SAME",
        )
        self.maxPool3d_2a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME")

        self.conv3d_2b_1x1 = Unit3Dpy(out_channels=64, in_channels=64, kernel_size=(1, 1, 1), padding="SAME")
        self.conv3d_2c_3x3 = Unit3Dpy(out_channels=192, in_channels=64, kernel_size=(3, 3, 3), padding="SAME")
        self.maxPool3d_3a_3x3 = MaxPool3dTFPadding(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding="SAME")

        self.mixed_3b = Mixed(192, [64, 96, 128, 16, 32, 32])
        self.mixed_3c = Mixed(256, [128, 128, 192, 32, 96, 64])

        self.maxPool3d_4a_3x3 = MaxPool3dTFPadding(kernel_size=(3, 3, 3), stride=(2, 2, 2), padding="SAME")

        self.mixed_4b = Mixed(480, [192, 96, 208, 16, 48, 64])
        self.mixed_4c = Mixed(512, [160, 112, 224, 24, 64, 64])
        self.mixed_4d = Mixed(512, [128, 128, 256, 24, 64, 64])
        self.mixed_4e = Mixed(512, [112, 144, 288, 32, 64, 64])
        self.mixed_4f = Mixed(528, [256, 160, 320, 32, 128, 128])

        self.maxPool3d_5a_2x2 = MaxPool3dTFPadding(kernel_size=(2, 2, 2), stride=(2, 2, 2), padding="SAME")

        self.mixed_5b = Mixed(832, [256, 160, 320, 32, 128, 128])
        self.mixed_5c = Mixed(832, [384, 192, 384, 48, 128, 128])

        self.pool = nn.AdaptiveAvgPool3d(1)
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, inp):
        if self.stem is not None:
            out = self.stem(inp) 
        else:
            out = self.conv3d_1a_7x7(inp)
        out = self.maxPool3d_2a_3x3(out)
        out = self.conv3d_2b_1x1(out)
        out = self.conv3d_2c_3x3(out)
        out = self.maxPool3d_3a_3x3(out)

        out = self.mixed_3b(out)
        out = self.mixed_3c(out)
        out = self.maxPool3d_4a_3x3(out)

        out = self.mixed_4b(out)
        out = self.mixed_4c(out)
        out = self.mixed_4d(out)
        out = self.mixed_4e(out)
        out = self.mixed_4f(out)

        out = self.maxPool3d_5a_2x2(out)
        out = self.mixed_5b(out)
        out = self.mixed_5c(out)

        out = self.pool(out).flatten(1)  # (B,1024)
        out = self.dropout(out)
        return out

def init_from_scratch(module: nn.Module):
    for m in module.modules():
        if isinstance(m, (nn.Conv3d, nn.Conv2d)):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.GroupNorm, nn.InstanceNorm3d, nn.InstanceNorm2d)):
            if getattr(m, "weight", None) is not None:
                nn.init.ones_(m.weight)
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)

class TwoStreamI3D_CLIP(nn.Module):
    """
    Top:  MHI clip  (B, C_mhi, 32, 224, 224)
    Bot:  Flow clip (B, 2,     128,112,112)

    Outputs:
      emb_top  (B,512)
      emb_bot  (B,512)
      emb_fuse (B,512)
    """
    def __init__(
        self,
        mhi_channels: int,
        second_channels: int = 2,
        embed_dim: int = 512,
        fuse: str = "avg_then_proj",
        dropout: float = 0.0,
        use_stems: bool = False,
        init_scratch: bool = True,
        compute_second_only: bool = False,
        use_nonlinear_projection: bool = False,
        active_branch: str = "both",
    ):        
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
        if use_stems:
            top_stem = InputStem3D(
                in_channels=mhi_channels,
                out_channels=64,
                norm="bn",
                kernel_size=(7, 7, 7),
                stride=(2, 2, 2),
                padding="SAME",
            )

            bot_stem = InputStem3D(
                in_channels=second_channels,
                out_channels=64,
                norm="in",                      # good default for low-channel motion streams
                kernel_size=(3, 7, 7),
                stride=(1, 2, 2),               # preserve time early
                padding="SAME",
            )
        else:
            top_stem = None
            bot_stem = None

        self.top = None
        self.bot = None
        self.proj_top = None
        self.proj_bot = None

        if self.has_top:
            self.top = I3DFeature(in_channels=mhi_channels, dropout_prob=dropout, stem=top_stem)
        if self.has_bot:
            self.bot = I3DFeature(in_channels=second_channels, dropout_prob=dropout, stem=bot_stem)

        if use_nonlinear_projection:
            if self.has_top:
                self.proj_top = MLPProjector(in_dim=1024, hidden_dim=2048, out_dim=embed_dim, dropout=dropout)
            if self.has_bot:
                self.proj_bot = MLPProjector(in_dim=1024, hidden_dim=2048, out_dim=embed_dim, dropout=dropout)
        else:
            if self.has_top:
                self.proj_top = nn.Linear(1024, embed_dim)
            if self.has_bot:
                self.proj_bot = nn.Linear(1024, embed_dim)

        self.fuse = fuse
        if fuse == "concat":
            if use_nonlinear_projection:
                self.proj_fuse = MLPProjector(in_dim=embed_dim * 2, hidden_dim=embed_dim * 2, out_dim=embed_dim)
            else:
                self.proj_fuse = nn.Linear(embed_dim * 2, embed_dim)
        elif fuse == "avg_then_proj":
            self.proj_fuse = nn.Linear(embed_dim, embed_dim)
        else:
            raise ValueError("fuse must be one of: concat, avg_then_proj")
        
        if init_scratch:
            init_from_scratch(self)


    def forward(self, mhi_bcthw, flow_bcthw):
        et = None
        eb = None

        if self.has_top:
            if mhi_bcthw is None:
                raise ValueError("active_branch requires first/top branch, but mhi input is None")
            ft = self.top(mhi_bcthw)    # (B,1024)
            et = self.proj_top(ft)      # (B,512)

        if self.has_bot:
            if flow_bcthw is None:
                raise ValueError("active_branch requires second/bot branch, but flow input is None")
            fb = self.bot(flow_bcthw)   # (B,1024)
            eb = self.proj_bot(fb)      # (B,512)

        if et is None and eb is None:
            raise RuntimeError("Model has no active branches. Set active_branch to both, first, or second.")
        if et is None:
            et = torch.zeros_like(eb)
        if eb is None:
            eb = torch.zeros_like(et)


        if self.fuse == "concat":
            ef = self.proj_fuse(torch.cat([et, eb], dim=-1))
        else:
            ef = self.proj_fuse(0.5 * (et + eb))

        return {"emb_top": et, "emb_bot": eb, "emb_fuse": ef}


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ----------------------------
    # Hard-coded settings
    # ----------------------------
    batch = 16

    mhi = torch.randn(16, 1, 32, 224, 224, device=device, dtype=torch.float32)
    flow = torch.randn(16, 2, 128, 160, 160, device=device, dtype=torch.float32)

    model = TwoStreamI3D_CLIP(
        mhi_channels=1,
        second_channels=2,
        embed_dim=512,
        fuse="concat",
        dropout=0.0,
        use_stems=True,
        init_scratch=True,
        compute_second_only=False,
        use_nonlinear_projection=True,
    ).to(device)

    model.eval()

    # ----------------------------
    # FLOP analysis (lazy import)
    # ----------------------------
    try:
        from fvcore.nn import FlopCountAnalysis

        flops = FlopCountAnalysis(model, (mhi, flow)).total()
        print("\n=== FLOP Analysis (forward only) ===")
        print(f"Total FLOPs: {flops / 1e9:.3f} GFLOPs\n")

    except ImportError:
        print("\n[INFO] fvcore not installed — skipping FLOP analysis.\n")
