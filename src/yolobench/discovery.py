"""Dynamic discovery of YOLO detection model variants supported by the
installed `ultralytics` package.

Never hardcodes a version list: newer families (e.g. YOLO26, YOLO27, ...)
must be picked up automatically as soon as the installed `ultralytics`
package supports them, without touching this code.
"""

from __future__ import annotations

import importlib.metadata
import re
from dataclasses import dataclass
from pathlib import Path

# Filenames that are detection-adjacent but not plain object detectors are
# excluded by pattern, not by an allowlist of "supported" families, so new
# detector families are picked up automatically.
_EXCLUDE_SUFFIXES = ("-seg", "-cls", "-pose", "-obb", "-world", "-worldv2", "-oiv7")
_EXCLUDE_PREFIXES = ("sam", "sam2", "mobile_sam", "rtdetr", "yoloe", "fastsam", "yolo_nas")

_FAMILY_VARIANT_RE = re.compile(r"^(yolo(?:v)?\d+)([a-z0-9]*)$")
_BASE_CFG_RE = re.compile(r"^yolo(v)?\d+$")


@dataclass(frozen=True)
class ModelSpec:
    family: str  # e.g. "yolov8", "yolo26"
    variant: str  # e.g. "n", "s", "m", "l", "x" (empty if unscaled)
    weights_name: str  # e.g. "yolov8n.pt"
    source: str  # "pretrained-checkpoint" | "yaml-cfg-only"


def get_ultralytics_version() -> str:
    return importlib.metadata.version("ultralytics")


def _is_excluded(stem: str) -> bool:
    lower = stem.lower()
    if any(lower.startswith(p) for p in _EXCLUDE_PREFIXES):
        return True
    return any(lower.endswith(s) for s in _EXCLUDE_SUFFIXES)


def _split_family_variant(stem: str) -> tuple[str, str]:
    match = _FAMILY_VARIANT_RE.match(stem)
    if not match:
        return stem, ""
    return match.group(1), match.group(2)


def discover_pretrained_checkpoints() -> list[ModelSpec]:
    """Primary strategy: read ultralytics' own list of known downloadable
    checkpoints. The attribute holding this list is not part of ultralytics'
    public API and has been renamed across releases, so a couple of
    candidate names are tried before giving up on this strategy.
    """
    try:
        from ultralytics.utils import downloads as ul_downloads
    except ImportError:
        return []

    names: list[str] = []
    for attr in ("GITHUB_ASSETS_NAMES", "GITHUB_ASSETS_STEMS"):
        value = getattr(ul_downloads, attr, None)
        if value:
            names = list(value)
            break

    specs: list[ModelSpec] = []
    seen: set[str] = set()
    for name in names:
        stem = Path(name).stem
        # Plain detectors are named "<family><scale>" (yolo11n, yolov5su).
        # Anything with an extra "-suffix" is a specialised head or variant
        # (-depth, -reid, -sem, -pose-p6, -grayscale, ...), not a COCO detector.
        if not _FAMILY_VARIANT_RE.match(stem.lower()) or _is_excluded(stem):
            continue
        weights_name = f"{stem}.pt"
        if weights_name in seen:
            continue
        seen.add(weights_name)
        family, variant = _split_family_variant(stem)
        specs.append(
            ModelSpec(family=family, variant=variant, weights_name=weights_name, source="pretrained-checkpoint")
        )
    return specs


def discover_from_cfg_yaml() -> list[ModelSpec]:
    """Fallback strategy, used only when `discover_pretrained_checkpoints`
    finds nothing (e.g. the internal attribute it relies on was renamed
    again in a future release). Scans ultralytics' bundled per-family model
    config YAMLs and assumes an "n" (nano) scale exists, since that is the
    convention for every family shipped so far.
    """
    try:
        import ultralytics
    except ImportError:
        return []

    cfg_dir = Path(ultralytics.__file__).parent / "cfg" / "models"
    if not cfg_dir.exists():
        return []

    specs: list[ModelSpec] = []
    seen_families: set[str] = set()
    for yaml_path in sorted(cfg_dir.glob("**/*.yaml")):
        stem = yaml_path.stem
        if _is_excluded(stem) or not _BASE_CFG_RE.match(stem) or stem in seen_families:
            continue
        seen_families.add(stem)
        specs.append(ModelSpec(family=stem, variant="n", weights_name=f"{stem}n.pt", source="yaml-cfg-only"))
    return specs


def discover_models() -> list[ModelSpec]:
    """Return all discovered detection model specs, deduplicated and sorted."""
    specs = discover_pretrained_checkpoints() or discover_from_cfg_yaml()

    seen: set[str] = set()
    unique: list[ModelSpec] = []
    for spec in specs:
        if spec.weights_name in seen:
            continue
        seen.add(spec.weights_name)
        unique.append(spec)
    return sorted(unique, key=lambda s: (s.family, s.variant))


# Scale names that differ from the usual n/s/m/l/x, mapped to the scale they
# correspond to so a `--sizes n` filter doesn't silently drop them:
# YOLOv9 names its smallest scale "t" (tiny) instead of "n".
_SIZE_ALIASES = {"t": "n"}


def normalized_size(variant: str) -> str:
    """Map a variant suffix to its plain scale letter for size filtering,
    e.g. "nu" (ultralytics' anchor-free YOLOv5/v3 retrains) -> "n",
    "t" (YOLOv9 tiny) -> "n". The weights filename is left untouched.
    """
    size = variant.lower()
    if len(size) > 1 and size.endswith("u"):
        size = size[:-1]
    return _SIZE_ALIASES.get(size, size)


def filter_specs(
    specs: list[ModelSpec],
    families: list[str] | None = None,
    sizes: list[str] | None = None,
) -> list[ModelSpec]:
    """Narrow the discovered list per `configs/benchmark.yaml` / CLI overrides."""
    result = specs
    if families:
        wanted = {f.lower() for f in families}
        result = [s for s in result if s.family.lower() in wanted]
    if sizes:
        wanted_sizes = {sz.lower() for sz in sizes}
        result = [s for s in result if normalized_size(s.variant) in wanted_sizes]
    return result
