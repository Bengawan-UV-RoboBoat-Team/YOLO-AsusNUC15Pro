"""Run inference (FPS/latency) + validation (mAP) for one (model, precision,
device) combination.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .devices import TARGET_DEVICES
from .discovery import ModelSpec, get_ultralytics_version


@dataclass
class RunResult:
    family: str
    variant: str
    precision: str
    device_key: str
    ov_device_name: str
    status: str  # "ok" | "export_failed" | "inference_failed"
    error: str | None
    fps_mean: float | None
    latency_ms_mean: float | None
    latency_ms_p95: float | None
    map50_95: float | None
    map50: float | None
    model_size_mb: float | None
    ultralytics_version: str
    timestamp: str

    def as_dict(self) -> dict:
        return asdict(self)


def _dir_size_mb(path: Path) -> float | None:
    if not path.exists():
        return None
    total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / (1024 * 1024), 3)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))
    return ordered[idx]


def run_one(
    spec: ModelSpec,
    ov_model_dir: Path,
    device_key: str,
    ov_device_name: str,
    sample_image: str,
    val_data_yaml: str,
    imgsz: int = 640,
    n_warmup: int = 10,
    n_timed: int = 100,
) -> RunResult:
    """Benchmark one exported OpenVINO model on one device. Never raises —
    failures are captured in the returned RunResult so one bad combo (e.g.
    an op the NPU plugin rejects) doesn't abort the rest of the sweep.
    """
    precision = ov_model_dir.name.replace("openvino_", "")
    base = dict(
        family=spec.family,
        variant=spec.variant,
        precision=precision,
        device_key=device_key,
        ov_device_name=ov_device_name,
        ultralytics_version=get_ultralytics_version(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

    try:
        from ultralytics import YOLO

        ov_device = TARGET_DEVICES[device_key]
        model = YOLO(str(ov_model_dir))

        for _ in range(n_warmup):
            model.predict(sample_image, device=ov_device, imgsz=imgsz, verbose=False)

        latencies_ms: list[float] = []
        for _ in range(n_timed):
            start = time.perf_counter()
            model.predict(sample_image, device=ov_device, imgsz=imgsz, verbose=False)
            latencies_ms.append((time.perf_counter() - start) * 1000)

        mean_latency_ms = sum(latencies_ms) / len(latencies_ms)
        fps_mean = 1000.0 / mean_latency_ms if mean_latency_ms else 0.0

        metrics = model.val(data=val_data_yaml, device=ov_device, imgsz=imgsz, verbose=False)

        return RunResult(
            **base,
            status="ok",
            error=None,
            fps_mean=round(fps_mean, 2),
            latency_ms_mean=round(mean_latency_ms, 3),
            latency_ms_p95=round(_percentile(latencies_ms, 95), 3),
            map50_95=round(float(metrics.box.map), 4),
            map50=round(float(metrics.box.map50), 4),
            model_size_mb=_dir_size_mb(ov_model_dir),
        )
    except Exception as exc:
        return RunResult(
            **base,
            status="inference_failed",
            error=str(exc),
            fps_mean=None,
            latency_ms_mean=None,
            latency_ms_p95=None,
            map50_95=None,
            map50=None,
            model_size_mb=_dir_size_mb(ov_model_dir),
        )
