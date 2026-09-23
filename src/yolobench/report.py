"""Aggregate RunResults into a CSV and a console pivot summary."""

from __future__ import annotations

import csv
from pathlib import Path

from .runner import RunResult

FIELDNAMES = [
    "family",
    "variant",
    "precision",
    "device_key",
    "ov_device_name",
    "status",
    "fps_mean",
    "latency_ms_mean",
    "latency_ms_p95",
    "map50_95",
    "map50",
    "model_size_mb",
    "ultralytics_version",
    "timestamp",
    "error",
]


def write_csv(results: list[RunResult], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        writer.writeheader()
        for result in results:
            row = result.as_dict()
            writer.writerow({k: row.get(k) for k in FIELDNAMES})


def print_summary(results: list[RunResult]) -> None:
    ok_results = [r for r in results if r.status == "ok"]
    if not ok_results:
        print("[report] no successful runs to summarize.")
        return

    try:
        import pandas as pd
    except ImportError:
        print("[report] pandas not installed, skipping pivot summary.")
        return

    df = pd.DataFrame([r.as_dict() for r in ok_results])
    df["model"] = df["family"] + df["variant"] + " (" + df["precision"] + ")"

    print("\n=== FPS (mean) ===")
    print(df.pivot_table(index="model", columns="device_key", values="fps_mean").round(1).to_string())

    print("\n=== mAP50-95 ===")
    print(df.pivot_table(index="model", columns="device_key", values="map50_95").round(3).to_string())

    failed = [r for r in results if r.status != "ok"]
    if failed:
        print(f"\n[report] {len(failed)} combo(s) failed - see the CSV 'error' column for details.")
