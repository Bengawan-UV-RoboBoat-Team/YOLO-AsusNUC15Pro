"""CLI entrypoint: discovery -> export -> run -> report.

Usage:
    python -m yolobench.benchmark
    python -m yolobench.benchmark --sizes n,s --precisions fp32,int8 --devices CPU,NPU
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from datetime import datetime, timezone
from pathlib import Path

import yaml

from . import devices as devices_mod
from . import discovery, report
from .export import OPENVINO_DIR_SUFFIX, ExportError, export_openvino
from .runner import RunResult, run_one

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "benchmark.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

# Give ultralytics a project-local settings file so datasets (coco128) land in
# data/ and val runs in results/runs/, without touching the user's global
# ultralytics settings. Must be set before ultralytics is first imported, and
# the directory must already exist or ultralytics falls back to /tmp.
_ULTRALYTICS_CONFIG_DIR = DATA_DIR / "ultralytics_config"
_ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(_ULTRALYTICS_CONFIG_DIR))


def _configure_ultralytics() -> None:
    import ultralytics.data.utils as ul_data_utils
    from ultralytics import settings

    settings.update({"datasets_dir": str(DATA_DIR), "runs_dir": str(RESULTS_DIR / "runs")})
    # ultralytics snapshots datasets_dir into a module constant at import
    # time, so on the very first run the update above only takes effect
    # from the next process on - patch the constant for this one too.
    ul_data_utils.DATASETS_DIR = DATA_DIR


def _load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark YOLO models across CPU/iGPU/NPU via OpenVINO.")
    parser.add_argument("--config", type=Path, default=CONFIG_PATH)
    parser.add_argument("--families", type=str, default=None, help="Comma-separated, e.g. yolov8,yolo11")
    parser.add_argument("--sizes", type=str, default=None, help="Comma-separated, e.g. n,s")
    parser.add_argument("--precisions", type=str, default=None, help="Comma-separated: fp32,fp16,int8")
    parser.add_argument("--devices", type=str, default=None, help="Comma-separated: CPU,GPU,NPU")
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--warmup-iters", type=int, default=None)
    parser.add_argument("--timed-iters", type=int, default=None)
    parser.add_argument("--force-export", action="store_true")
    return parser.parse_args()


def _sample_image() -> str:
    import ultralytics

    candidate = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    if candidate.exists():
        return str(candidate)
    raise FileNotFoundError(
        "Could not find ultralytics' bundled sample image (assets/bus.jpg) for timing runs."
    )


def _failed_result(
    spec: discovery.ModelSpec, precision: str, device_key: str, ov_device_name: str, status: str, error: str
) -> RunResult:
    return RunResult(
        family=spec.family,
        variant=spec.variant,
        precision=precision,
        device_key=device_key,
        ov_device_name=ov_device_name,
        status=status,
        error=error,
        fps_mean=None,
        latency_ms_mean=None,
        latency_ms_p95=None,
        map50_95=None,
        map50=None,
        model_size_mb=None,
        ultralytics_version=discovery.get_ultralytics_version(),
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _run_isolated(
    spec: discovery.ModelSpec, ov_dir: Path, device_key: str, ov_device_name: str, *args, **kwargs
) -> RunResult:
    """Run `run_one` in a fresh child process. OpenVINO GPU/NPU plugins can
    crash natively (segfault) on unsupported models, which no try/except can
    catch - isolating each combo turns that into a "crashed" row instead of
    killing the whole sweep. It also stops one run's device state (compiled
    model caches, memory) from leaking into the next measurement.
    """
    with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
        future = pool.submit(run_one, spec, ov_dir, device_key, ov_device_name, *args, **kwargs)
        try:
            return future.result()
        except BrokenProcessPool:
            precision = ov_dir.name.removesuffix(OPENVINO_DIR_SUFFIX)
            error = "process crashed (native OpenVINO/driver fault, e.g. segfault)"
            return _failed_result(spec, precision, device_key, ov_device_name, "crashed", error)


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)

    families = args.families.split(",") if args.families else cfg.get("families") or None
    sizes = args.sizes.split(",") if args.sizes else cfg.get("sizes") or ["n", "s"]
    precisions = args.precisions.split(",") if args.precisions else cfg.get("precisions") or ["fp32", "int8"]
    requested_devices = args.devices.split(",") if args.devices else cfg.get("devices") or ["CPU", "GPU", "NPU"]
    dataset = args.dataset or cfg.get("dataset") or "coco128.yaml"
    imgsz = args.imgsz or cfg.get("imgsz") or 640
    n_warmup = args.warmup_iters or cfg.get("warmup_iters") or 10
    n_timed = args.timed_iters or cfg.get("timed_iters") or 100
    force_export = args.force_export or bool(cfg.get("force_export"))

    _configure_ultralytics()
    print(f"[benchmark] ultralytics {discovery.get_ultralytics_version()}")
    all_specs = discovery.discover_models()
    specs = discovery.filter_specs(all_specs, families=families, sizes=sizes)
    print(f"[benchmark] discovered {len(all_specs)} model(s), running {len(specs)} after filtering:")
    for spec in specs:
        print(f"  - {spec.weights_name} ({spec.source})")
    skipped = [s.weights_name for s in all_specs if s not in specs]
    if skipped:
        print(f"[benchmark] not run (filtered out by families/sizes): {', '.join(skipped)}")

    detected = devices_mod.probe_openvino_devices()
    resolved_devices = devices_mod.resolve_run_devices(requested_devices)
    if not resolved_devices:
        print("[benchmark] no requested devices are available on this machine - nothing to run.")
        return

    sample_image = _sample_image()

    total_combos = len(specs) * len(precisions) * len(resolved_devices)
    combo_idx = 0
    results: list[RunResult] = []
    run_timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    raw_path = RESULTS_DIR / "raw" / f"{run_timestamp}.json"
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    def flush() -> None:
        with open(raw_path, "w", encoding="utf-8") as fh:
            json.dump([r.as_dict() for r in results], fh, indent=2)

    for spec in specs:
        for precision in precisions:
            try:
                ov_dir = export_openvino(spec, precision, imgsz=imgsz, calib_data=dataset, force=force_export)
            except ExportError as exc:
                combo_idx += len(resolved_devices)
                print(f"[{combo_idx}/{total_combos}] {spec.weights_name} / {precision}: EXPORT FAILED - {exc}")
                results.append(_failed_result(spec, precision, "N/A", "", "export_failed", str(exc)))
                flush()
                continue

            for device_key in resolved_devices:
                combo_idx += 1
                print(f"[{combo_idx}/{total_combos}] {spec.weights_name} / {precision} / {device_key} ...")
                result = _run_isolated(
                    spec,
                    ov_dir,
                    device_key,
                    detected.get(device_key, device_key),
                    sample_image,
                    dataset,
                    imgsz=imgsz,
                    n_warmup=n_warmup,
                    n_timed=n_timed,
                )
                status_note = "ok" if result.status == "ok" else f"FAILED - {result.error}"
                print(f"    -> {status_note}")
                results.append(result)
                flush()

    report.write_csv(results, RESULTS_DIR / "benchmark_summary.csv")
    report.print_summary(results)
    print(f"\n[benchmark] done. raw results: {raw_path}")
    print(f"[benchmark] summary: {RESULTS_DIR / 'benchmark_summary.csv'}")


if __name__ == "__main__":
    main()
