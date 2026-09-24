"""Validation datasets that don't overlap with the pretrained models'
training data.

`coco128.yaml` points both `train:` and `val:` at 128 images from COCO
train2017, the split every pretrained checkpoint was trained on, so mAP
measured on it rewards memorisation rather than detection quality. The
"coco-val<N>" datasets here are a fixed, evenly spaced N-image subset of
COCO val2017 (never used for training), so mAP is comparable across model
families.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_COCO_VAL_RE = re.compile(r"^coco-val(\d+)$")
_COCO_VAL_SIZE = 5000
_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"  # ~1 GB, 5k images


def _coco_names() -> dict:
    """Class names from ultralytics' bundled coco.yaml, so they always match
    the installed version instead of being hardcoded here."""
    from ultralytics.utils.checks import check_yaml

    with open(check_yaml("coco.yaml"), "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)["names"]


def _download_coco_val(coco_dir: Path) -> None:
    """Fetch only the val2017 images and the box labels, not the 19 GB of
    train2017 images that ultralytics' own coco.yaml would download."""
    from ultralytics.utils import ASSETS_URL
    from ultralytics.utils.downloads import download

    if not (coco_dir / "labels" / "val2017").is_dir():
        print("[dataset] downloading COCO labels")
        # The archive's top-level folder is "coco/", so extract next to it.
        download([f"{ASSETS_URL}/coco2017labels.zip"], dir=coco_dir.parent, delete=True)

    images_dir = coco_dir / "images" / "val2017"
    if len(list(images_dir.glob("*.jpg"))) < _COCO_VAL_SIZE:
        print("[dataset] downloading COCO val2017 images (~1 GB)")
        download([_VAL_IMAGES_URL], dir=coco_dir / "images", delete=True)


def ensure_coco_val_subset(data_dir: Path, n: int = 500) -> str:
    """Make sure a fixed n-image subset of COCO val2017 exists under
    `data_dir` and return the path of its dataset yaml. Downloads only
    when the images are missing; the yaml is rewritten on every call so its
    absolute path stays valid after data/ is copied to another machine.
    """
    if not 0 < n <= _COCO_VAL_SIZE:
        raise ValueError(f"coco-val subset size must be 1-{_COCO_VAL_SIZE}, got {n}")

    coco_dir = data_dir / "coco"
    _download_coco_val(coco_dir)

    images = sorted((coco_dir / "images" / "val2017").glob("*.jpg"))
    step = len(images) // n
    subset = images[::step][:n]

    list_file = coco_dir / f"val2017_{n}.txt"
    list_file.write_text("".join(f"./images/val2017/{p.name}\n" for p in subset), encoding="utf-8")

    yaml_path = data_dir / f"coco-val{n}.yaml"
    # ultralytics requires both keys; train is never used by the benchmark.
    dataset = {"path": str(coco_dir), "train": list_file.name, "val": list_file.name, "names": _coco_names()}
    with open(yaml_path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(dataset, fh, sort_keys=False)
    return str(yaml_path)


def resolve_dataset(name: str, data_dir: Path) -> str:
    """Turn a dataset name from the config/CLI into something ultralytics
    can load: "coco-val<N>" is built on demand, anything else (a bundled
    yaml such as coco128.yaml, or a path to a custom yaml) passes through."""
    match = _COCO_VAL_RE.match(name)
    if match:
        return ensure_coco_val_subset(data_dir, int(match.group(1)))
    return name
