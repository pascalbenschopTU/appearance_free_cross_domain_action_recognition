import argparse
import ast
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple

try:
    import tomllib
except ImportError:
    try:
        import tomli as tomllib  # type: ignore
    except ImportError:
        tomllib = None

try:
    import yaml  # type: ignore
except ImportError:
    yaml = None


CONFIG_META_KEYS = {
    "_self_",
    "defaults",
    "overriden_values",
    "selected_option",
}

COMMA_SEPARATED_KEYS = {
    "head_weights",
    "mhi_windows",
    "unfreeze_modules",
    "use_heads",
}


def _add_config_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("Config")
    group.add_argument(
        "--config",
        type=str,
        action="append",
        default=None,
        help=(
            "Optional config file (.json; .toml also works via stdlib or the built-in fallback parser; "
            ".yaml/.yml when PyYAML is installed). "
            "Later files override earlier ones; CLI flags override config values."
        ),
    )


def _parse_simple_toml_value(raw_value: str) -> Any:
    value = raw_value.strip()
    if not value:
        return ""
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    if value.startswith(("'", '"')):
        return ast.literal_eval(value)
    if value.startswith("["):
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            return ast.literal_eval(value.replace("true", "True").replace("false", "False"))
    if value.startswith("{"):
        return ast.literal_eval(value.replace("true", "True").replace("false", "False"))
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _simple_toml_value_complete(raw_value: str) -> bool:
    bracket_depth = 0
    brace_depth = 0
    quote_char = None
    escaped = False

    for ch in raw_value:
        if quote_char is not None:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == quote_char:
                quote_char = None
            continue

        if ch in ('"', "'"):
            quote_char = ch
        elif ch == "[":
            bracket_depth += 1
        elif ch == "]":
            bracket_depth = max(0, bracket_depth - 1)
        elif ch == "{":
            brace_depth += 1
        elif ch == "}":
            brace_depth = max(0, brace_depth - 1)

    return quote_char is None and bracket_depth == 0 and brace_depth == 0


def _load_simple_toml(text: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {}
    current: Dict[str, Any] = data
    pending_key: Optional[str] = None
    pending_value_lines = []

    for line in text.splitlines():
        stripped = line.strip()

        if pending_key is not None:
            if stripped and not stripped.startswith("#"):
                pending_value_lines.append(stripped)
            raw_value = "\n".join(pending_value_lines)
            if _simple_toml_value_complete(raw_value):
                current[pending_key] = _parse_simple_toml_value(raw_value)
                pending_key = None
                pending_value_lines = []
            continue

        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section_path = [part.strip() for part in stripped[1:-1].split(".") if part.strip()]
            current = data
            for section in section_path:
                current = current.setdefault(section, {})
                if not isinstance(current, dict):
                    raise ValueError(f"Invalid TOML section path: {stripped}")
            continue
        if "=" not in stripped:
            raise ValueError(f"Invalid TOML line: {line}")
        key, raw_value = stripped.split("=", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if _simple_toml_value_complete(raw_value):
            current[key] = _parse_simple_toml_value(raw_value)
        else:
            pending_key = key
            pending_value_lines = [raw_value]

    if pending_key is not None:
        raise ValueError(f"Unterminated TOML value for key '{pending_key}'")

    return data


def _load_config_file(path: str) -> Dict[str, Any]:
    config_path = Path(path).expanduser()
    text = config_path.read_text(encoding="utf-8")
    suffix = config_path.suffix.lower()

    if suffix == ".json":
        data = json.loads(text)
    elif suffix == ".toml":
        data = tomllib.loads(text) if tomllib is not None else _load_simple_toml(text)
    elif suffix in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError(
                "YAML config support requires PyYAML. Use TOML/JSON or install PyYAML."
            )
        data = yaml.safe_load(text)
    else:
        errors = []
        for loader_name, loader in (
            ("json", json.loads),
            ("toml", None if tomllib is None else tomllib.loads),
            ("yaml", None if yaml is None else yaml.safe_load),
        ):
            if loader is None:
                continue
            try:
                data = loader(text)
                break
            except Exception as exc:
                errors.append(f"{loader_name}: {exc}")
        else:
            joined = "; ".join(errors) if errors else "no loaders available"
            raise RuntimeError(f"Could not parse config file {path!r}: {joined}")

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a mapping/object at the top level: {path}")
    return data


def _iter_config_leaves(node: Any, path: Tuple[str, ...] = ()) -> Iterable[Tuple[Tuple[str, ...], Any]]:
    if isinstance(node, dict):
        for key, value in node.items():
            key_str = str(key)
            if key_str in CONFIG_META_KEYS:
                continue
            yield from _iter_config_leaves(value, path + (key_str,))
        return
    if not path:
        raise ValueError("Config file must contain at least one key.")
    yield path, node


def _normalize_config_value(action: argparse.Action, dest: str, value: Any) -> Any:
    if dest in COMMA_SEPARATED_KEYS and isinstance(value, (list, tuple)):
        return ",".join(str(item) for item in value)
    if action.nargs in ("*", "+") and value is not None and not isinstance(value, (list, tuple)):
        return [value]
    if action.nargs in ("*", "+") and isinstance(value, tuple):
        return list(value)
    return value


def apply_config_defaults(
    parser: argparse.ArgumentParser,
    config_paths: Optional[Sequence[str]],
) -> None:
    if not config_paths:
        return

    actions = {
        action.dest: action
        for action in parser._actions
        if action.dest not in {argparse.SUPPRESS, "help"}
    }

    defaults: Dict[str, Any] = {}
    unknown_keys = []
    for config_path in config_paths:
        file_defaults: Dict[str, str] = {}
        data = _load_config_file(config_path)
        for key_path, value in _iter_config_leaves(data):
            dotted = ".".join(key_path)
            leaf = key_path[-1]
            if leaf not in actions:
                unknown_keys.append(f"{config_path}: {dotted}")
                continue
            previous = file_defaults.get(leaf)
            if previous is not None and previous != dotted:
                raise ValueError(
                    f"Ambiguous config keys in {config_path!r}: both '{previous}' and '{dotted}' "
                    f"map to argparse destination '{leaf}'."
                )
            file_defaults[leaf] = dotted
            defaults[leaf] = _normalize_config_value(actions[leaf], leaf, value)

    if unknown_keys:
        preview = "\n".join(unknown_keys[:20])
        if len(unknown_keys) > 20:
            preview += f"\n... and {len(unknown_keys) - 20} more"
        raise ValueError(f"Unknown config keys:\n{preview}")

    parser.set_defaults(**defaults)
    for dest in defaults:
        action = actions.get(dest)
        if action is not None and getattr(action, "required", False):
            action.required = False


def parse_args_with_config(
    parser: argparse.ArgumentParser,
    argv: Optional[Sequence[str]] = None,
) -> argparse.Namespace:
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, action="append", default=None)
    pre_args, _ = pre_parser.parse_known_args(argv)
    apply_config_defaults(parser, pre_args.config)
    return parser.parse_args(argv)


def build_train_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    data = parser.add_argument_group("Data")
    data.add_argument("--root_dir", type=str, required=True)
    data.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument("--val_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument("--val_root_dir", type=str, default="")
    data.add_argument("--val_manifest", type=str, default="", help="Validation split manifest (file or glob).")
    data.add_argument("--val_class_id_to_label_csv", type=str, default="")
    data.add_argument("--val_class_text_json", type=str, default="")

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=224)
    motion.add_argument("--mhi_frames", type=int, default=32)
    motion.add_argument("--flow_frames", type=int, default=128, help="frames to produce 128 flows")
    motion.add_argument("--flow_hw", type=int, default=112)
    motion.add_argument("--second_type", type=str, default="flow")
    motion.add_argument("--rgb_frames", type=int, default=64)
    motion.add_argument(
        "--rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument("--mhi_windows", type=str, default="15", help="comma list, e.g. 5,25")
    motion.add_argument("--diff_threshold", type=float, default=15.0)
    motion.add_argument("--flow_max_disp", type=float, default=20.0)
    motion.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback"])
    motion.add_argument("--fb_pyr_scale", type=float, default=0.5)
    motion.add_argument("--fb_levels", type=int, default=3)
    motion.add_argument("--fb_winsize", type=int, default=15)
    motion.add_argument("--fb_iterations", type=int, default=3)
    motion.add_argument("--fb_poly_n", type=int, default=5)
    motion.add_argument("--fb_poly_sigma", type=float, default=1.2)
    motion.add_argument("--fb_flags", type=int, default=0)
    motion.add_argument("--motion_img_resize", type=int, default=None)
    motion.add_argument("--motion_flow_resize", type=int, default=None)
    motion.add_argument(
        "--motion_resize_mode",
        type=str,
        default="square",
        choices=["square", "short_side"],
    )
    motion.add_argument(
        "--motion_eval_crop_mode",
        type=str,
        default="none",
        choices=["none", "random", "center", "motion"],
    )
    motion.add_argument(
        "--motion_eval_num_views",
        type=int,
        default=1,
        help="Number of spatial motion views per video for validation. >1 uses fixed multi-crop anchors.",
    )
    motion.add_argument(
        "--motion_spatial_crop",
        type=str,
        default="random",
        choices=["random", "motion"],
    )

    model = parser.add_argument_group("Model")
    model.add_argument("--embed_dim", type=int, default=512)
    model.add_argument("--fuse", type=str, default="avg_then_proj", choices=["avg_then_proj", "concat"])
    model.add_argument("--model", type=str, default="i3d", choices=["i3d", "x3d"])
    model.add_argument("--x3d_variant", type=str.upper, default="XS", choices=["XS", "S", "M", "L"])
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--use_stems", action="store_true")
    model.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    model.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)
    model.add_argument(
        "--use_projection",
        action="store_true",
        help="Enable separate fused clip/CE heads (LayerNorm+Linear for CLIP, Dropout+Linear for CE).",
    )
    model.add_argument(
        "--dual_projection_heads",
        action="store_true",
        help="Use separate fused projection heads for CLIP-style CE and embedding alignment losses.",
    )
    model.add_argument("--use_nonlinear_projection", action="store_true", help=argparse.SUPPRESS)

    augmentation = parser.add_argument_group("Augmentation")
    augmentation.add_argument("--probability_hflip", type=float, default=0.5)
    augmentation.add_argument(
        "--max_probability_drop_frame",
        type=float,
        default=0.0,
        help="max probability for zeroing frames",
    )
    augmentation.add_argument("--probability_affine", type=float, default=0.0, help="rotate,translate,scale,shear")
    augmentation.add_argument("--label_smoothing", type=float, default=0.0)
    augmentation.add_argument("--temporal_mixup_prob", type=float, default=0.0)
    augmentation.add_argument("--temporal_mixup_y_min", type=float, default=0.35)
    augmentation.add_argument("--temporal_mixup_y_max", type=float, default=0.65)
    augmentation.add_argument(
        "--lambda_rep_mix",
        type=float,
        default=0.0,
        help="Weight for representation-space mix consistency loss.",
    )
    augmentation.add_argument(
        "--rep_mix_alpha",
        type=float,
        default=0.4,
        help="Beta(alpha, alpha) parameter for representation-space mix.",
    )
    augmentation.add_argument(
        "--rep_mix_semantic",
        action="store_true",
        help="Select representation-mix partners from semantically close classes within the current batch.",
    )
    augmentation.add_argument(
        "--rep_mix_semantic_topk",
        type=int,
        default=3,
        help="Randomly choose among top-k semantic partners found in-batch.",
    )
    augmentation.add_argument(
        "--rep_mix_semantic_min_sim",
        type=float,
        default=-1.0,
        help="Minimum cosine similarity for semantic partner candidates; values <= -1 disable filtering.",
    )

    text = parser.add_argument_group("Text Supervision")
    text.add_argument("--class_text_json", type=str, default="")
    text.add_argument(
        "--text_supervision_mode",
        type=str,
        default="class_proto",
        choices=["class_proto", "desc_soft_margin", "class_multi_positive"],
    )
    text.add_argument(
        "--description_match_csv",
        type=str,
        default="",
        help="CSV with per-video matched descriptions for desc_soft_margin supervision or matched_desc embed targets.",
    )
    text.add_argument(
        "--embed_target_mode",
        type=str,
        default="class_proto",
        choices=["class_proto", "matched_desc"],
        help="Target source for embedding alignment losses (L2/cosine).",
    )
    text.add_argument(
        "--embed_target_label_mix_weight",
        type=float,
        default=0.0,
        help="Optional weight for mixing a class-label CLIP embedding into matched_desc embedding targets.",
    )
    text.add_argument(
        "--embed_target_label_template",
        type=str,
        default="a video of {}",
        help="Template used to build class-label embeddings when --embed_target_label_mix_weight > 0.",
    )
    text.add_argument("--text_bank_backend", type=str, default="clip", choices=["clip", "precomputed"])
    text.add_argument("--precomputed_text_embeddings", type=str, default="")
    text.add_argument("--precomputed_text_index", type=str, default="")
    text.add_argument("--precomputed_text_key", type=str, default="")
    text.add_argument(
        "--apply_templates_to_class_texts",
        dest="apply_templates_to_class_texts",
        action="store_true",
        help="Apply CLIP templates to class labels/custom class texts.",
    )
    text.add_argument(
        "--no_apply_templates_to_class_texts",
        dest="apply_templates_to_class_texts",
        action="store_false",
        help="Disable templates for class labels/custom class texts.",
    )
    text.add_argument(
        "--apply_templates_to_class_descriptions",
        action="store_true",
        help="Also apply CLIP templates to long-form descriptions (default: disabled).",
    )
    text.add_argument(
        "--class_text_label_weight",
        type=float,
        default=0.5,
        help=(
            "Label-anchor weight when class labels and descriptions are combined. "
            "For class_proto this is alpha*t_label + (1-alpha)*t_desc; "
            "for class_multi_positive this assigns alpha to the class label and spreads (1-alpha) across descriptions."
        ),
    )
    text.add_argument(
        "--text_adapter",
        type=str,
        default="none",
        choices=["none", "linear", "mlp"],
        help="Optional residual adapter applied to frozen text embeddings before loss/eval.",
    )
    text.add_argument(
        "--lambda_clip_ce",
        type=float,
        default=1.0,
        help="Weight for CLIP-style CE over text bank similarities.",
    )
    text.add_argument(
        "--lambda_embed_cos",
        type=float,
        default=0.0,
        help="Weight for cosine embedding alignment against target embeddings from --embed_target_mode.",
    )
    text.add_argument(
        "--lambda_ce",
        type=float,
        default=0.0,
        help="Weight for auxiliary CE loss using a linear head on fused embeddings.",
    )
    text.add_argument(
        "--unfreeze_logit_scale",
        action="store_true",
        help="Freeze logit_scale parameter while keeping it in the optimizer param list for checkpoint compatibility.",
    )
    parser.set_defaults(apply_templates_to_class_texts=True)

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument("--batch_size", type=int, default=16)
    optimization.add_argument("--epochs", type=int, default=40)
    optimization.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "sgd"])
    optimization.add_argument("--lr", type=float, default=2e-4)
    optimization.add_argument("--weight_decay", type=float, default=1e-4)
    optimization.add_argument("--sgd_momentum", type=float, default=0.9)
    optimization.add_argument("--sgd_nesterov", action="store_true")
    optimization.add_argument("--warmup_steps", type=int, default=4000)
    optimization.add_argument("--min_lr", type=float, default=1e-6)

    validation = parser.add_argument_group("Validation")
    validation.add_argument("--val_every", type=int, default=1, help="Run validation every N epochs (0 disables).")
    validation.add_argument(
        "--val_samples_per_class",
        type=int,
        default=0,
        help="If >0, subsample validation set to at most this many samples per class.",
    )
    validation.add_argument(
        "--val_subset_seed",
        type=int,
        default=0,
        help="Seed for deterministic validation per-class subsampling.",
    )

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--num_workers", type=int, default=16)
    runtime.add_argument("--log_every", type=int, default=100)
    runtime.add_argument("--save_every", type=int, default=2000)
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--out_dir", type=str, default="out/train")
    runtime.add_argument(
        "--clip_cache_dir",
        type=str,
        default="",
        help="Directory for CLIP model downloads. Defaults to out/clip shared across runs.",
    )
    runtime.add_argument("--tb_dir", type=str, default="runs")
    runtime.add_argument("--ckpt_dir", type=str, default="checkpoints")
    runtime.add_argument("--device", type=str, default=default_device)
    return parser


def parse_train_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_train_parser(default_device)
    return parse_args_with_config(parser, argv)


def build_privacy_pa_hmdb51_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    this_dir = Path(__file__).resolve().parent
    workspace_root = this_dir.parent.parent

    data = parser.add_argument_group("Data")
    data.add_argument("--root_dir", type=str, default=str(workspace_root / "datasets" / "hmdb51"))
    data.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb", "mhi", "flow"])
    data.add_argument(
        "--privacy_attr_dir",
        type=str,
        default=str(this_dir / "privacy" / "data" / "pa_hmdb51" / "PrivacyAttributes"),
    )
    data.add_argument(
        "--hmdb_val_manifest_dir",
        type=str,
        default=str(this_dir / "tc-clip" / "datasets_splits" / "hmdb_splits"),
    )
    data.add_argument(
        "--hmdb_label_csv",
        type=str,
        default=str(this_dir / "tc-clip" / "labels" / "hmdb_51_labels.csv"),
    )
    data.add_argument("--out_dir", type=str, default=str(this_dir / "privacy" / "out" / "pa_hmdb51_privacy_cv"))
    data.add_argument(
        "--attributes",
        type=str,
        default="all",
        help="Comma-separated list from: gender,skin_color,face,nudity,relationship or 'all'.",
    )
    data.add_argument("--prepare_only", action="store_true")

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=224)
    motion.add_argument("--mhi_frames", type=int, default=32)
    motion.add_argument("--flow_frames", type=int, default=128)
    motion.add_argument("--flow_hw", type=int, default=112)
    motion.add_argument("--mhi_windows", type=str, default="25")
    motion.add_argument("--rgb_frames", type=int, default=64)
    motion.add_argument(
        "--rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument("--diff_threshold", type=float, default=15.0)
    motion.add_argument("--flow_max_disp", type=float, default=20.0)
    motion.add_argument("--flow_normalize", action="store_true")
    motion.add_argument("--no_flow_normalize", dest="flow_normalize", action="store_false")
    parser.set_defaults(flow_normalize=True)
    motion.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback"])
    motion.add_argument("--fb_pyr_scale", type=float, default=0.5)
    motion.add_argument("--fb_levels", type=int, default=3)
    motion.add_argument("--fb_winsize", type=int, default=15)
    motion.add_argument("--fb_iterations", type=int, default=3)
    motion.add_argument("--fb_poly_n", type=int, default=5)
    motion.add_argument("--fb_poly_sigma", type=float, default=1.2)
    motion.add_argument("--fb_flags", type=int, default=0)
    motion.add_argument("--motion_img_resize", type=int, default=256)
    motion.add_argument("--motion_flow_resize", type=int, default=128)
    motion.add_argument(
        "--motion_resize_mode",
        type=str,
        default="square",
        choices=["square", "short_side"],
    )
    motion.add_argument(
        "--motion_crop_mode",
        type=str,
        default="none",
        choices=["none", "random", "center"],
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model_backbone",
        type=str,
        default="i3d",
        choices=["i3d", "resnet18", "resnet50"],
        help="Encoder backbone. ResNet variants treat rgb/mhi/flow as image-like sequences.",
    )
    model.add_argument("--embed_dim", type=int, default=512)
    model.add_argument("--fuse", type=str, default="avg_then_proj", choices=["avg_then_proj", "concat"])
    model.add_argument("--dropout", type=float, default=0.0)
    model.add_argument("--head_dropout", type=float, default=0.0)
    model.add_argument("--use_stems", action="store_true")
    model.add_argument("--active_branch", type=str, default="both", choices=["both", "first", "second"])
    model.add_argument("--class_weight_mode", type=str, default="effective_sample_count", choices=["none", "inverse_freq", "sqrt_inverse_freq", "effective_sample_count", "effective_num"])
    model.add_argument("--class_aware_sampling", action="store_true", default=False,
                       help="Use WeightedRandomSampler to oversample minority-class videos (overrides RepeatedVideoTemporalSampler).")
    model.add_argument("--resnet_imagenet_pretrained", action="store_true")
    model.add_argument("--no_resnet_imagenet_pretrained", dest="resnet_imagenet_pretrained", action="store_false")
    parser.set_defaults(resnet_imagenet_pretrained=True)
    model.add_argument("--resnet_temporal_samples", type=int, default=4)
    model.add_argument("--resnet_temporal_pool", type=str, default="avg", choices=["avg", "max"])

    optimization = parser.add_argument_group("Optimization")
    optimization.add_argument("--batch_size", type=int, default=16)
    optimization.add_argument("--epochs", type=int, default=10)
    optimization.add_argument("--lr", type=float, default=5e-4)
    optimization.add_argument("--min_lr", type=float, default=1e-5)
    optimization.add_argument("--weight_decay", type=float, default=1e-4)
    optimization.add_argument("--warmup_steps", type=int, default=20)

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--pretrained_ckpt", type=str, default="")
    runtime.add_argument("--resume", type=str, default="")
    runtime.add_argument("--device", type=str, default=default_device)
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--num_workers", type=int, default=16)
    runtime.add_argument("--print_every", type=int, default=20)
    return parser


def parse_privacy_pa_hmdb51_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_privacy_pa_hmdb51_parser(default_device)
    return parse_args_with_config(parser, argv)


def build_eval_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    data = parser.add_argument_group("Data")
    data.add_argument("--root_dir", type=str, required=True)
    data.add_argument("--ckpt", type=str, required=True)
    data.add_argument("--out_dir", type=str, default="eval_out")
    data.add_argument("--manifests", type=str, nargs="*", default=None, help="evaluation splits")
    data.add_argument("--input_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument(
        "--motion_data_source",
        type=str,
        default="video",
        choices=["video", "zstd"],
        help="For motion evaluation: 'video' computes motion on the fly, 'zstd' loads precomputed motion tensors.",
    )
    data.add_argument("--class_id_to_label_csv", type=str, default=None)

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--device", type=str, default=default_device)
    runtime.add_argument("--batch_size", type=int, default=8)
    runtime.add_argument("--num_workers", type=int, default=0)
    runtime.add_argument(
        "--summary_only",
        action="store_true",
        help="Skip confusion matrices and per-class artifacts; write summary JSON only.",
    )
    runtime.add_argument(
        "--clip_cache_dir",
        type=str,
        default="",
        help="Directory for CLIP model downloads. Defaults to out/clip shared across runs.",
    )

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=224)
    motion.add_argument("--mhi_frames", type=int, default=32)
    motion.add_argument("--flow_frames", type=int, default=128)
    motion.add_argument("--flow_hw", type=int, default=112)
    motion.add_argument("--mhi_windows", type=str, default="15")
    motion.add_argument("--diff_threshold", type=float, default=15.0)
    motion.add_argument("--flow_max_disp", type=float, default=20.0)
    motion.add_argument("--model_rgb_frames", type=int, default=64)
    motion.add_argument(
        "--model_rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--model_rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument(
        "--flow_backend",
        type=str,
        default="farneback",
        choices=["farneback", "raft_large"],
        help="Flow extractor for on-the-fly evaluation.",
    )
    motion.add_argument(
        "--raft_flow_clip",
        type=float,
        default=1.0,
        help="Clip RAFT flow to [-x, x] before model input (default: 1.0, matching RAFT zst conversion). Set <=0 to disable.",
    )
    motion.add_argument("--raft_amp", action="store_true", default=True, help="Use AMP for RAFT inference on CUDA.")
    motion.add_argument("--no_raft_amp", action="store_false", dest="raft_amp", help="Disable AMP for RAFT inference.")
    motion.add_argument(
        "--roi_mode",
        type=str,
        default="none",
        choices=["none", "largest_motion", "yolo_person"],
        help="Optional ROI pre-crop mode for VideoMotionDataset",
    )
    motion.add_argument("--roi_stride", type=int, default=3, help="Frame stride for ROI prepass")
    motion.add_argument(
        "--motion_roi_threshold",
        type=float,
        default=None,
        help="Threshold for largest_motion ROI (default: --diff_threshold)",
    )
    motion.add_argument("--motion_roi_min_area", type=int, default=64, help="Min CC area for largest_motion ROI")
    motion.add_argument("--yolo_model", type=str, default="yolo11n.pt", help="YOLO model name/path (ultralytics)")
    motion.add_argument("--yolo_conf", type=float, default=0.25, help="YOLO confidence threshold")
    motion.add_argument("--yolo_device", type=str, default=None, help="YOLO device, e.g. cpu or 0")
    motion.add_argument("--fb_pyr_scale", type=float, default=0.5)
    motion.add_argument("--fb_levels", type=int, default=3)
    motion.add_argument("--fb_winsize", type=int, default=15)
    motion.add_argument("--fb_iterations", type=int, default=3)
    motion.add_argument("--fb_poly_n", type=int, default=5)
    motion.add_argument("--fb_poly_sigma", type=float, default=1.2)
    motion.add_argument("--fb_flags", type=int, default=0)
    motion.add_argument("--motion_img_resize", type=int, default=256, help="None keeps the target-size legacy path.")
    motion.add_argument("--motion_flow_resize", type=int, default=128, help="None keeps the target-size legacy path.")
    motion.add_argument(
        "--motion_resize_mode",
        type=str,
        default="short_side",
        choices=["square", "short_side"],
        help="Spatial resize policy.",
    )
    motion.add_argument(
        "--motion_eval_crop_mode",
        type=str,
        default="center",
        choices=["none", "random", "center", "motion"],
        help="Spatial crop mode for evaluation.",
    )
    motion.add_argument(
        "--motion_eval_num_views",
        type=int,
        default=1,
        help="Number of spatial motion views per video for evaluation. >1 uses fixed multi-crop anchors.",
    )

    text = parser.add_argument_group("Text / Ensembling")
    text.add_argument("--text_bank_backend", type=str, default="clip", choices=["clip", "precomputed"])
    text.add_argument("--precomputed_text_embeddings", type=str, default="")
    text.add_argument("--precomputed_text_index", type=str, default="")
    text.add_argument("--precomputed_text_key", type=str, default="")
    text.add_argument("--class_text_json", type=str, default="")
    text.add_argument(
        "--text_supervision_mode",
        type=str,
        default="",
        choices=["", "class_proto", "desc_soft_margin", "class_multi_positive"],
        help="Optional eval-time override for text supervision aggregation. Empty uses checkpoint setting.",
    )
    text.add_argument("--use_heads", type=str, default="fuse")
    text.add_argument("--head_weights", type=str, default="1.0")
    text.add_argument("--logit_scale", type=float, default=0.0)
    text.add_argument(
        "--active_branch",
        type=str,
        default=None,
        choices=["both", "first", "second"],
        help="None -> use checkpoint setting",
    )
    text.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)
    text.add_argument("--no_clip", action="store_true", help="Skip CLIP RGB embeddings; evaluate the model branch only.")
    text.add_argument("--no_rgb", dest="no_clip", action="store_true", help=argparse.SUPPRESS)
    text.add_argument("--rgb_frames", type=int, default=1)
    text.add_argument("--rgb_sampling", type=str, default="center", choices=["center", "uniform", "random"])
    text.add_argument("--rgb_weight", type=float, default=0.5)
    text.add_argument("--clip_vision_scale", type=float, default=0.0)
    return parser


def parse_eval_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_eval_parser(default_device)
    return parse_args_with_config(parser, argv)


def build_finetune_parser(default_device: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    _add_config_args(parser)

    data = parser.add_argument_group("Data")
    data.add_argument("-r", "--root_dir", type=str, required=True)
    data.add_argument("-m", "--manifest", type=str, default=None, help="ONE split manifest (file or glob). Optional.")
    data.add_argument("-c", "--class_id_to_label_csv", type=str, default=None)
    data.add_argument("--train_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument("--val_modality", type=str, default="motion", choices=["motion", "rgb"])
    data.add_argument(
        "--motion_data_source",
        type=str,
        default="zstd",
        choices=["zstd", "video"],
        help="For --train_modality motion: 'zstd' loads precomputed motion tensors, 'video' computes MHI+flow on-the-fly.",
    )
    data.add_argument("--val_root_dir", type=str, default=None)
    data.add_argument("--val_manifest", type=str, default=None, help="ONE validation split manifest (file or glob). Optional.")
    data.add_argument("--val_class_id_to_label_csv", type=str, default=None)
    data.add_argument("--val_class_text_json", type=str, default=None, help="Optional JSON mapping validation classes to prompt lists.")
    data.add_argument("--train_class_text_json", type=str, default=None, help="Optional JSON mapping training classes to prompt lists/descriptions.")
    data.add_argument(
        "--text_bank_backend",
        type=str,
        default="clip",
        choices=["clip", "precomputed"],
        help="Text embedding backend for class bank: CLIP encoder or precomputed embeddings.",
    )
    data.add_argument(
        "--precomputed_text_embeddings",
        type=str,
        default=None,
        help="Path to precomputed text embeddings .npy (e.g., sentence_transformer_embeddings.npy).",
    )
    data.add_argument(
        "--precomputed_text_index",
        type=str,
        default=None,
        help="Path to index JSON for precomputed text embeddings.",
    )
    data.add_argument(
        "--precomputed_text_key",
        type=str,
        default=None,
        help="Dataset key in precomputed index JSON (e.g., kinetics_400_llm_labels).",
    )
    data.add_argument(
        "--val_subset_size",
        type=int,
        default=400,
        help="Use a fixed random subset for validation if >0; <=0 means full split.",
    )
    data.add_argument(
        "--val_samples_per_class",
        type=int,
        default=0,
        help="If >0, keep at most this many validation samples per class before any global subset.",
    )
    data.add_argument(
        "--val_subset_seed",
        type=int,
        default=0,
        help="Seed for deterministic validation subset selection.",
    )

    text = parser.add_argument_group("Text")
    text.add_argument(
        "--apply_templates_to_class_texts",
        dest="apply_templates_to_class_texts",
        action="store_true",
        help="Apply CLIP templates to class labels/custom class texts.",
    )
    text.add_argument(
        "--no_apply_templates_to_class_texts",
        dest="apply_templates_to_class_texts",
        action="store_false",
        help="Disable templates for class labels/custom class texts.",
    )
    text.add_argument(
        "--apply_templates_to_class_descriptions",
        action="store_true",
        help="Also apply CLIP templates to long-form descriptions (default: disabled).",
    )
    text.add_argument(
        "--class_text_label_weight",
        type=float,
        default=0.5,
        help=(
            "Label-anchor weight when class labels and descriptions are combined."
        ),
    )
    text.add_argument(
        "--text_adapter",
        type=str,
        default="none",
        choices=["none", "linear", "mlp"],
        help="Optional residual adapter applied to frozen text embeddings before loss/eval.",
    )
    text.add_argument(
        "--text_supervision_mode",
        type=str,
        default="class_label",
        choices=["class_label", "class_averaged", "class_multi_positive"],
        help=(
            "How to use text descriptions during training. "
            "'class_label': single class-name embedding (ignores train_class_text_json descriptions). "
            "'class_averaged': weighted average of label + descriptions (alpha=class_text_label_weight). "
            "'class_multi_positive': multi-positive contrastive loss over label + all description embeddings."
        ),
    )
    text.add_argument(
        "--lambda_clip_ce",
        type=float,
        default=1.0,
        help="Weight for CLIP-style CE over text bank similarities.",
    )
    text.add_argument(
        "--lambda_ce",
        type=float,
        default=0.0,
        help="Weight for auxiliary CE loss using a linear head on fused embeddings.",
    )
    parser.set_defaults(apply_templates_to_class_texts=True)

    pretrained = parser.add_argument_group("Pretrained")
    pretrained.add_argument(
        "-p",
        "--pretrained_ckpt",
        type=str,
        default=None,
        help="checkpoint path OR directory (optional; omit for scratch training)",
    )

    motion = parser.add_argument_group("Motion")
    motion.add_argument("--img_size", type=int, default=None)
    motion.add_argument("--mhi_frames", type=int, default=None)
    motion.add_argument("--flow_frames", type=int, default=None)
    motion.add_argument("--flow_hw", type=int, default=None)
    motion.add_argument("--mhi_windows", type=str, default=None, help="comma list, e.g. 5,25 (None -> inherit)")
    motion.add_argument("--diff_threshold", type=float, default=15.0, help="diff threshold for mhi")
    motion.add_argument("--flow_max_disp", type=float, default=None, help="Clip flow to [-x, x] before model input.")
    motion.add_argument("--flow_normalize", action="store_true", default=True, help="Normalize flow by --flow_max_disp.")
    motion.add_argument("--no_flow_normalize", action="store_false", dest="flow_normalize")
    motion.add_argument("--flow_backend", type=str, default="farneback", choices=["farneback"])
    motion.add_argument("--fb_pyr_scale", type=float, default=None)
    motion.add_argument("--fb_levels", type=int, default=None)
    motion.add_argument("--fb_winsize", type=int, default=None)
    motion.add_argument("--fb_iterations", type=int, default=None)
    motion.add_argument("--fb_poly_n", type=int, default=None)
    motion.add_argument("--fb_poly_sigma", type=float, default=None)
    motion.add_argument("--fb_flags", type=int, default=None)
    motion.add_argument("--motion_img_resize", type=int, default=256, help="None keeps the target-size legacy path.")
    motion.add_argument("--motion_flow_resize", type=int, default=128, help="None keeps the target-size legacy path.")
    motion.add_argument(
        "--motion_resize_mode",
        type=str,
        default="short_side",
        choices=["square", "short_side"],
        help="Spatial resize policy.",
    )
    motion.add_argument(
        "--motion_train_crop_mode",
        type=str,
        default="random",
        choices=["none", "random", "center"],
        help="Spatial crop mode for training.",
    )
    motion.add_argument(
        "--motion_eval_crop_mode",
        type=str,
        default="center",
        choices=["none", "random", "center", "motion"],
        help="Spatial crop mode for evaluation.",
    )
    motion.add_argument("--roi_mode", type=str, default="none", choices=["none", "largest_motion", "yolo_person"])
    motion.add_argument("--roi_stride", type=int, default=3)
    motion.add_argument("--motion_roi_threshold", type=float, default=None)
    motion.add_argument("--motion_roi_min_area", type=int, default=64)
    motion.add_argument("--yolo_model", type=str, default="yolo11n.pt")
    motion.add_argument("--yolo_conf", type=float, default=0.25)
    motion.add_argument("--yolo_device", type=str, default=None)
    motion.add_argument("--rgb_frames", type=int, default=64)
    motion.add_argument(
        "--rgb_sampling",
        type=str,
        default="uniform",
        choices=["uniform", "center", "random"],
    )
    motion.add_argument("--rgb_norm", type=str, default="i3d", choices=["i3d", "clip", "none"])
    motion.add_argument("--motion_spatial_crop", type=str, default="random", choices=["random", "motion"])
    motion.add_argument(
        "--p_hflip",
        "--probability_hflip",
        dest="p_hflip",
        type=float,
        default=0.5,
        help="Probability of applying horizontal flip augmentation during motion finetuning.",
    )
    motion.add_argument(
        "--p_affine",
        "--probability_affine",
        dest="p_affine",
        type=float,
        default=0.0,
        help="Probability of applying geometric affine augmentation during motion finetuning.",
    )
    motion.add_argument(
        "--color_jitter",
        type=float,
        default=0.0,
        help="Probability of applying ColorJitter to RGB frames during training (0.0 = off, 0.8 = TC-CLIP-like).",
    )
    motion.add_argument(
        "--motion_noise_std",
        type=float,
        default=0.0,
        help="Std of Gaussian noise added to MHI/flow tensors during motion training (0.0 = off).",
    )

    model = parser.add_argument_group("Model")
    model.add_argument(
        "--model",
        type=str,
        default=None,
        choices=["i3d", "x3d"],
        help="None -> inherit from pretrained checkpoint",
    )
    model.add_argument("--embed_dim", type=int, default=None)
    model.add_argument("--fuse", type=str, default=None, choices=[None, "avg_then_proj", "concat"])
    model.add_argument("--dropout", type=float, default=None)
    model.add_argument(
        "--active_branch",
        type=str,
        default=None,
        choices=["both", "first", "second"],
        help="None -> inherit from pretrained checkpoint",
    )
    model.add_argument("--compute_second_only", action="store_true", help=argparse.SUPPRESS)
    model.add_argument(
        "--use_projection",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    model.add_argument(
        "--dual_projection_heads",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    model.add_argument("--use_nonlinear_projection", action="store_true", help=argparse.SUPPRESS)
    model.add_argument("--lambda_align", type=float, default=0.0)
    model.add_argument(
        "--lambda_cls",
        type=float,
        default=0.0,
        help="Weight for auxiliary CLS-token classification loss when model provides logits_cls.",
    )

    finetune = parser.add_argument_group("Finetune")
    finetune.add_argument("--freeze_backbone", action="store_true", default=True)
    finetune.add_argument("--no_freeze_backbone", action="store_false", dest="freeze_backbone")
    finetune.add_argument("--unfreeze_modules", type=str, default="", help="e.g. 'mixed_5b,mixed_5c'")
    finetune.add_argument(
        "--finetune_head_mode",
        type=str,
        default="legacy",
        choices=["legacy", "language", "class", "both"],
        help=(
            "legacy keeps the original finetune behavior. "
            "language/class/both run a head-only ablation that updates only the selected prediction head(s)."
        ),
    )
    finetune.add_argument(
        "--freeze_bn_stats",
        action="store_true",
        default=True,
        help="Keep BatchNorm layers in eval mode (no running-stat updates).",
    )
    finetune.add_argument(
        "--no_freeze_bn_stats",
        action="store_false",
        dest="freeze_bn_stats",
        help="Allow BatchNorm running stats to adapt during finetuning.",
    )
    finetune.add_argument("--batch_size", type=int, default=16)
    finetune.add_argument("--epochs", type=int, default=50)
    finetune.add_argument("--lr", type=float, default=2e-4)
    finetune.add_argument("--weight_decay", type=float, default=1e-4)
    finetune.add_argument("--warmup_steps", type=int, default=1000)
    finetune.add_argument("--min_lr", type=float, default=1e-6)
    finetune.add_argument("--label_smoothing", type=float, default=0.0)
    finetune.add_argument("--mixup_alpha", type=float, default=0.0)
    finetune.add_argument("--mixup_prob", type=float, default=0.0)
    finetune.add_argument("--temporal_mixup_prob", type=float, default=0.0)
    finetune.add_argument("--temporal_mixup_y_min", type=float, default=0.35)
    finetune.add_argument("--temporal_mixup_y_max", type=float, default=0.65)
    finetune.add_argument("--lambda_rep_mix", type=float, default=0.0, help="Weight for representation-space mix consistency loss.")
    finetune.add_argument("--rep_mix_alpha", type=float, default=0.4, help="Beta(alpha, alpha) parameter for representation-space mix.")
    finetune.add_argument("--rep_mix_semantic", action="store_true", help="Select representation-mix partners from semantically close classes within the current batch.")
    finetune.add_argument("--rep_mix_semantic_topk", type=int, default=3, help="Randomly choose among top-k semantic partners found in-batch.")
    finetune.add_argument("--rep_mix_semantic_min_sim", type=float, default=-1.0, help="Minimum cosine similarity for semantic partner candidates; values <= -1 disable filtering.")

    runtime = parser.add_argument_group("Runtime")
    runtime.add_argument("--num_workers", type=int, default=16)
    runtime.add_argument("--log_every", type=int, default=100)
    runtime.add_argument("--save_every", type=int, default=200)
    runtime.add_argument(
        "--checkpoint_mode",
        type=str,
        default="best",
        choices=["best", "latest", "final"],
        help="'best' keeps the best checkpoint according to the active criterion; 'latest' overwrites a single rolling checkpoint; 'final' saves only once after the last epoch.",
    )
    runtime.add_argument("--max_updates", type=int, default=0, help="Stop after this many optimizer updates (0 disables).")
    runtime.add_argument("--seed", type=int, default=0)
    runtime.add_argument("--out_dir", type=str, default="out/finetune")
    runtime.add_argument(
        "--clip_cache_dir",
        type=str,
        default="",
        help="Directory for CLIP model downloads. Defaults to out/clip shared across runs.",
    )
    runtime.add_argument("--tb_dir", type=str, default="runs")
    runtime.add_argument("--ckpt_dir", type=str, default="checkpoints")
    runtime.add_argument("--eval_dir", type=str, default="eval_out")
    runtime.add_argument("--val_skip_epochs", type=int, default=5, help="Skip validation for the first N epochs.")
    runtime.add_argument("--val_every", type=int, default=1, help="Validation interval after the skip.")
    runtime.add_argument("--early_stop_patience", type=int, default=0, help="Stop if validation top1 does not improve for N validation checks (0 disables).")
    runtime.add_argument("--early_stop_min_delta", type=float, default=0.0, help="Minimum top1 improvement required to reset early stopping counter.")
    runtime.add_argument("--device", type=str, default=default_device)
    return parser


def parse_finetune_args(
    argv: Optional[Sequence[str]] = None,
    *,
    default_device: str,
) -> argparse.Namespace:
    parser = build_finetune_parser(default_device)
    return parse_args_with_config(parser, argv)
