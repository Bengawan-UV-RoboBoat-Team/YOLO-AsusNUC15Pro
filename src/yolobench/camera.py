"""Live camera test: run one or more camera streams through exported
OpenVINO models at the same time and check whether they keep up.

Usage:
    python -m yolobench.camera                  # interactive: pick camera, model, precision, device
    python -m yolobench.camera --list           # show which models are already exported
    python -m yolobench.camera --stream 0:yolo26s:int8:GPU
    python -m yolobench.camera --stream 0:yolo26s:int8:GPU --stream 2:yolo26s:fp32:NPU --track --duration 600

A stream is SOURCE:MODEL:PRECISION:DEVICE. SOURCE is a camera index (0, 1,
...), a video file or a stream URL; it is split from the right, so URLs
with colons (rtsp://host:554/...) work.

Unlike the benchmark (one model, one image, idle machine), every stream
here runs concurrently on live frames, so the numbers include the load the
streams put on each other, on the CPU and on the device.
"""

from __future__ import annotations

import argparse
import csv
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from . import devices as devices_mod
from . import discovery
from .benchmark import RESULTS_DIR, _configure_ultralytics
from .devices import TARGET_DEVICES
from .export import MODELS_DIR, OPENVINO_DIR_SUFFIX, ExportError, export_openvino

PRECISIONS = ("fp32", "fp16", "int8")
CAMERA_RESULTS_DIR = RESULTS_DIR / "camera"
# Only needed when a requested int8 model hasn't been exported yet.
CALIB_DATASET = "coco128.yaml"
N_WARMUP = 5
STATUS_EVERY_S = 5.0


@dataclass(frozen=True)
class StreamConfig:
    source: str
    model: str  # e.g. "yolo26s"
    precision: str
    device: str  # "CPU" | "GPU" | "NPU"

    @property
    def name(self) -> str:
        return f"{self.source} | {self.model} {self.precision} @ {self.device}"


def parse_stream(text: str) -> StreamConfig:
    parts = text.rsplit(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(f"expected SOURCE:MODEL:PRECISION:DEVICE, got '{text}'")
    source, model, precision, device = parts
    precision, device = precision.lower(), device.upper()
    if precision not in PRECISIONS:
        raise argparse.ArgumentTypeError(f"precision must be one of {PRECISIONS}, got '{precision}'")
    if device not in TARGET_DEVICES:
        raise argparse.ArgumentTypeError(f"device must be one of {list(TARGET_DEVICES)}, got '{device}'")
    return StreamConfig(source, model.lower(), precision, device)


_SIZE_ORDER = "nsmlx"


def exported_models() -> dict[str, list[str]]:
    """{model name: [precisions]} for every OpenVINO export already in
    models/, ordered by family and then size (n, s, m, l, x)."""

    def order(ov_dir: Path) -> tuple:
        size = discovery.normalized_size(ov_dir.parent.name)
        return ov_dir.parent.parent.name, _SIZE_ORDER.find(size) % 10, ov_dir.name

    found: dict[str, list[str]] = {}
    for ov_dir in sorted(MODELS_DIR.glob(f"*/*/*{OPENVINO_DIR_SUFFIX}"), key=order):
        family, variant = ov_dir.parent.parent.name, ov_dir.parent.name
        name = family + ("" if variant == "base" else variant)
        found.setdefault(name, []).append(ov_dir.name.removesuffix(OPENVINO_DIR_SUFFIX))
    return found


def resolve_model_dir(model: str, precision: str, imgsz: int) -> Path:
    """Return the OpenVINO export for `model` at `precision`, exporting it
    first (like the benchmark does) if it isn't in models/ yet."""
    specs = {Path(s.weights_name).stem: s for s in discovery.discover_models()}
    if model not in specs:
        raise SystemExit(f"[camera] unknown model '{model}'. Known: {', '.join(sorted(specs))}")
    return export_openvino(specs[model], precision, imgsz=imgsz, calib_data=CALIB_DATASET)


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(pct / 100 * (len(ordered) - 1)))]


class Capture(threading.Thread):
    """Reads frames as fast as the source delivers them and keeps only the
    newest one, so a slow model skips frames instead of falling behind
    (which is what a robot wants: always act on the latest frame)."""

    def __init__(
        self, source: str, width: int | None, height: int | None, fps: float | None, fourcc: str | None = None
    ) -> None:
        super().__init__(daemon=True)
        import cv2

        self.cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
        if not self.cap.isOpened():
            raise SystemExit(f"[camera] can't open source '{source}'")
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        # The pixel format must be set before the size: many USB cameras
        # only reach 30 FPS at 720p+ in MJPG, not in the default YUYV.
        if fourcc:
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, fps)
        # A video file is read as fast as possible unless paced to its own
        # frame rate, which is what makes it behave like a live camera.
        self.is_file = not source.isdigit() and Path(source).exists()
        self.file_period = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30.0) if self.is_file else 0.0

        self.lock = threading.Lock()
        self.frame = None
        self.frame_id = 0
        self.frame_time = 0.0
        self.ended = threading.Event()

    def run(self) -> None:
        next_due = time.perf_counter()
        while not self.ended.is_set():
            ok, frame = self.cap.read()
            if not ok:
                self.ended.set()
                break
            if self.is_file:
                next_due += self.file_period
                time.sleep(max(0.0, next_due - time.perf_counter()))
            with self.lock:
                self.frame, self.frame_id, self.frame_time = frame, self.frame_id + 1, time.perf_counter()
        self.cap.release()

    def latest(self):
        with self.lock:
            return self.frame, self.frame_id, self.frame_time


@dataclass
class StreamStats:
    captured_start: int = 0
    captured: int = 0
    processed: int = 0
    skipped: int = 0
    infer_ms: list[float] = field(default_factory=list)
    e2e_ms: list[float] = field(default_factory=list)  # frame grabbed -> result ready
    start: float = 0.0
    end: float = 0.0


class Worker(threading.Thread):
    """Runs one stream's model on the newest frame of its capture, forever."""

    def __init__(self, cfg: StreamConfig, capture: Capture, ov_dir: Path, imgsz: int, track: bool, show: bool):
        super().__init__(daemon=True)
        self.cfg, self.capture, self.ov_dir = cfg, capture, ov_dir
        self.imgsz, self.track, self.show = imgsz, track, show
        self.stats = StreamStats()
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.error: str | None = None
        self.annotated = None

    def _infer(self, model, frame):
        kwargs = dict(device=TARGET_DEVICES[self.cfg.device], imgsz=self.imgsz, verbose=False)
        if self.track:
            return model.track(frame, persist=True, **kwargs)[0]
        return model.predict(frame, **kwargs)[0]

    def run(self) -> None:
        try:
            from ultralytics import YOLO

            model = YOLO(str(self.ov_dir), task="detect")
            frame = None
            while frame is None and not self.capture.ended.is_set():
                frame, last_id, _ = self.capture.latest()
                time.sleep(0.01)
            if frame is None:
                raise RuntimeError("source delivered no frames")
            for _ in range(N_WARMUP):
                self._infer(model, frame)
        except Exception as exc:
            self.error = str(exc)
            self.ready.set()
            return

        self.stats.start = time.perf_counter()
        self.stats.captured_start = self.capture.latest()[1]
        self.ready.set()
        while not self.stop.is_set() and not self.capture.ended.is_set():
            frame, frame_id, frame_time = self.capture.latest()
            if frame_id == last_id:
                time.sleep(0.001)
                continue
            self.stats.skipped += frame_id - last_id - 1
            last_id = frame_id

            t0 = time.perf_counter()
            result = self._infer(model, frame)
            t1 = time.perf_counter()
            self.stats.infer_ms.append((t1 - t0) * 1000)
            self.stats.e2e_ms.append((t1 - frame_time) * 1000)
            self.stats.processed += 1
            if self.show:
                self.annotated = result.plot()
        self.stats.end = time.perf_counter()
        self.stats.captured = self.capture.latest()[1] - self.stats.captured_start


def _cpu_temp_c() -> float | None:
    import psutil

    try:
        temps = psutil.sensors_temperatures()
    except (AttributeError, OSError):  # not available on Windows
        return None
    readings = [t.current for t in temps.get("coretemp", []) if t.current]
    return max(readings) if readings else None


def _summarize(worker: Worker, cam_fps: float) -> dict:
    s = worker.stats
    elapsed = max(1e-9, (s.end or time.perf_counter()) - s.start)
    source_fps = (s.captured or (worker.capture.latest()[1] - s.captured_start)) / elapsed
    target_fps = cam_fps or source_fps
    frame_budget_ms = 1000.0 / target_fps if target_fps else 0.0
    processed_fps = s.processed / elapsed
    skipped_pct = 100.0 * s.skipped / max(1, s.skipped + s.processed)
    infer_p95 = _percentile(s.infer_ms, 95)
    keeps_up = skipped_pct <= 5.0 and infer_p95 <= frame_budget_ms
    return {
        "stream": worker.cfg.name,
        "source": worker.cfg.source,
        "model": worker.cfg.model,
        "precision": worker.cfg.precision,
        "device": worker.cfg.device,
        "track": worker.track,
        "seconds": round(elapsed, 1),
        "source_fps": round(source_fps, 1),
        "processed_fps": round(processed_fps, 1),
        "skipped_pct": round(skipped_pct, 1),
        "infer_ms_mean": round(sum(s.infer_ms) / len(s.infer_ms), 1) if s.infer_ms else None,
        "infer_ms_p95": round(infer_p95, 1),
        "infer_ms_p99": round(_percentile(s.infer_ms, 99), 1),
        "e2e_ms_p95": round(_percentile(s.e2e_ms, 95), 1),
        "e2e_ms_p99": round(_percentile(s.e2e_ms, 99), 1),
        "frame_budget_ms": round(frame_budget_ms, 1),
        "verdict": "OK" if keeps_up else "TOO SLOW",
    }


def _ask(prompt: str, options: list[str], default: str | None = None) -> str:
    for i, opt in enumerate(options, 1):
        print(f"  {i:>2}. {opt}")
    hint = f" [{default}]" if default else ""
    while True:
        answer = input(f"{prompt}{hint}: ").strip() or (default or "")
        if answer.isdigit() and 1 <= int(answer) <= len(options):
            return options[int(answer) - 1]
        if answer in options:
            return answer
        print("  pick a number from the list")


def interactive_streams(detected: list[str]) -> list[StreamConfig]:
    exported = exported_models()
    if not exported:
        raise SystemExit("[camera] no exported models in models/ - run the benchmark with --prepare first")
    streams: list[StreamConfig] = []
    while True:
        print(f"\n=== stream {len(streams) + 1} ===")
        source = input("Camera index, video file or URL [0]: ").strip() or "0"
        print("Model:")
        model = _ask("Model", list(exported))
        print("Precision:")
        precision = _ask("Precision", [p for p in PRECISIONS if p in exported[model]])
        print("Device:")
        device = _ask("Device", detected, default=detected[0] if len(detected) == 1 else None)
        streams.append(StreamConfig(source, model, precision, device))
        if input("Add another stream? [y/N]: ").strip().lower() != "y":
            return streams


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Live camera test for exported YOLO OpenVINO models.")
    parser.add_argument(
        "--stream",
        action="append",
        type=parse_stream,
        help="SOURCE:MODEL:PRECISION:DEVICE, e.g. 0:yolo26s:int8:GPU. Repeat for more cameras. "
        "Without --stream you are asked interactively.",
    )
    parser.add_argument("--list", action="store_true", help="List exported models and exit")
    parser.add_argument("--duration", type=float, default=0, help="Stop after N seconds (default: until q / Ctrl+C)")
    parser.add_argument("--track", action="store_true", help="Run ByteTrack tracking on top of detection")
    parser.add_argument("--no-show", action="store_true", help="Don't open preview windows (headless / SSH)")
    parser.add_argument("--imgsz", type=int, default=640, help="Model input size, must match the export")
    parser.add_argument("--cam-fps", type=float, default=30, help="Camera frame rate to request and judge against")
    parser.add_argument("--width", type=int, default=None, help="Camera frame width to request")
    parser.add_argument("--height", type=int, default=None, help="Camera frame height to request")
    parser.add_argument("--fourcc", type=str, default=None, help="Camera pixel format to request, e.g. MJPG")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_ultralytics()

    if args.list:
        for model, precisions in exported_models().items():
            print(f"  {model:<10} {', '.join(precisions)}")
        return

    detected = devices_mod.resolve_run_devices(list(TARGET_DEVICES))
    streams = args.stream or interactive_streams(detected)
    missing = sorted({s.device for s in streams} - set(detected))
    if missing:
        raise SystemExit(f"[camera] device(s) not available on this machine: {', '.join(missing)}")

    show = not args.no_show
    workers: list[Worker] = []
    captures: dict[str, Capture] = {}
    for cfg in streams:
        try:
            ov_dir = resolve_model_dir(cfg.model, cfg.precision, args.imgsz)
        except ExportError as exc:
            raise SystemExit(f"[camera] {exc}") from exc
        # Two streams on the same source share one capture (e.g. the same
        # camera through a GPU model and an NPU model).
        if cfg.source not in captures:
            captures[cfg.source] = Capture(cfg.source, args.width, args.height, args.cam_fps, args.fourcc)
            captures[cfg.source].start()
        workers.append(Worker(cfg, captures[cfg.source], ov_dir, args.imgsz, args.track, show))

    print(f"[camera] loading and warming up {len(workers)} stream(s) ...")
    for w in workers:
        w.start()
    for w in workers:
        w.ready.wait()
        if w.error:
            raise SystemExit(f"[camera] {w.cfg.name}: {w.error}")
    print("[camera] running - press q in a preview window or Ctrl+C to stop\n")

    import cv2
    import psutil

    psutil.cpu_percent(interval=None)
    cpu_samples: list[float] = []
    temp_max: float | None = None
    started = last_status = time.perf_counter()
    try:
        while any(w.is_alive() for w in workers):
            now = time.perf_counter()
            if args.duration and now - started >= args.duration:
                break
            if show:
                for i, w in enumerate(workers):
                    if w.annotated is not None:
                        cv2.imshow(f"[{i + 1}] {w.cfg.name}", w.annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            else:
                time.sleep(0.05)
            if now - last_status >= STATUS_EVERY_S:
                last_status = now
                cpu = psutil.cpu_percent(interval=None)
                cpu_samples.append(cpu)
                temp = _cpu_temp_c()
                if temp is not None:
                    temp_max = max(temp_max or temp, temp)
                line = " | ".join(
                    f"[{i + 1}] {len(w.stats.infer_ms) / max(1e-9, now - w.stats.start):.1f} fps "
                    f"p95 {_percentile(w.stats.infer_ms[-300:], 95):.1f} ms"
                    for i, w in enumerate(workers)
                )
                temp_text = f", CPU temp {temp:.0f}°C" if temp is not None else ""
                print(f"[{now - started:6.0f}s] {line} | CPU {cpu:.0f}%{temp_text}")
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers:
            w.stop.set()
        for w in workers:
            w.join(timeout=5)
        for c in captures.values():
            c.ended.set()
        if show:
            cv2.destroyAllWindows()

    rows = [_summarize(w, args.cam_fps) for w in workers]
    cpu_mean = round(sum(cpu_samples) / len(cpu_samples), 1) if cpu_samples else None
    print("\n=== camera test summary ===")
    for i, r in enumerate(rows, 1):
        print(
            f"[{i}] {r['stream']}{' + track' if r['track'] else ''}\n"
            f"    {r['processed_fps']} of {r['source_fps']} fps processed, {r['skipped_pct']}% frames skipped\n"
            f"    inference mean {r['infer_ms_mean']} / p95 {r['infer_ms_p95']} / p99 {r['infer_ms_p99']} ms "
            f"(budget {r['frame_budget_ms']} ms per frame)\n"
            f"    frame-to-result p95 {r['e2e_ms_p95']} / p99 {r['e2e_ms_p99']} ms\n"
            f"    -> {r['verdict']}"
        )
    print(f"CPU load mean {cpu_mean}%" + (f", max CPU temp {temp_max:.0f}°C" if temp_max is not None else ""))

    CAMERA_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out = CAMERA_RESULTS_DIR / f"{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.csv"
    with open(out, "w", newline="", encoding="utf-8") as fh:
        extra = {"cpu_load_mean": cpu_mean, "cpu_temp_max": temp_max, "streams_total": len(rows)}
        writer = csv.DictWriter(fh, fieldnames=[*rows[0], *extra])
        writer.writeheader()
        for r in rows:
            writer.writerow({**r, **extra})
    print(f"[camera] saved {out}")


if __name__ == "__main__":
    main()
