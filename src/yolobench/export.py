"""Export .pt weights to OpenVINO IR, cached per (model, precision)."""

from __future__ import annotations

import shutil
from pathlib import Path

from .discovery import ModelSpec

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
# Pretrained .pt checkpoints are downloaded here (ultralytics fetches a
# missing checkpoint to whatever path it is given) instead of into the cwd.
WEIGHTS_DIR = MODELS_DIR / "weights"


class ExportError(RuntimeError):
    """Raised when a model can't be exported, so callers can record a
    structured failure row instead of crashing the whole benchmark run."""


# Ultralytics only recognises an OpenVINO IR directory as a loadable model
# when its name ends in "_openvino_model", so the precision goes in front.
OPENVINO_DIR_SUFFIX = "_openvino_model"

# `quantize` export argument per precision (ultralytics >= 8.4 replaced the
# old `half` / `int8` flags with this single argument).
_QUANTIZE = {"fp32": 32, "fp16": 16, "int8": 8}


def _export_dir(spec: ModelSpec, precision: str) -> Path:
    return MODELS_DIR / spec.family / (spec.variant or "base") / f"{precision}{OPENVINO_DIR_SUFFIX}"


def export_openvino(
    spec: ModelSpec,
    precision: str,
    imgsz: int = 640,
    calib_data: str | None = None,
    force: bool = False,
) -> Path:
    """Export `spec`'s pretrained weights to an OpenVINO IR directory for the
    given precision ("fp32" | "fp16" | "int8"), reusing a cached export when
    one already exists (export is the slow step and the same IR is reused
    across all benchmarked devices).
    """
    out_dir = _export_dir(spec, precision)
    if out_dir.exists() and not force:
        return out_dir

    if precision == "int8" and not calib_data:
        raise ExportError(
            f"{spec.weights_name}: int8 export requires calib_data (a dataset yaml) for NNCF calibration"
        )

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise ExportError(f"{spec.weights_name}: ultralytics is not installed ({exc})") from exc

    try:
        model = YOLO(str(WEIGHTS_DIR / spec.weights_name))
        export_kwargs = {"format": "openvino", "imgsz": imgsz, "quantize": _QUANTIZE[precision]}
        if precision == "int8":
            export_kwargs["data"] = calib_data
        exported_path = Path(model.export(**export_kwargs))
    except ExportError:
        raise
    except Exception as exc:
        raise ExportError(f"{spec.weights_name} [{precision}]: export failed: {exc}") from exc

    out_dir.parent.mkdir(parents=True, exist_ok=True)
    if exported_path != out_dir:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        exported_path.rename(out_dir)

    return out_dir
