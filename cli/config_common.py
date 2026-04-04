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

__all__ = [
    "COMMA_SEPARATED_KEYS",
    "CONFIG_META_KEYS",
    "_add_config_args",
    "_iter_config_leaves",
    "_load_config_file",
    "_load_simple_toml",
    "_normalize_config_value",
    "_parse_simple_toml_value",
    "_simple_toml_value_complete",
    "apply_config_defaults",
    "parse_args_with_config",
]
