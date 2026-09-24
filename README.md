# YOLO Benchmark: Asus NUC 15 Pro

**English** | [Bahasa Indonesia](README.id.md)

Benchmarks every YOLO model version supported by the `ultralytics` package
(detected automatically, from YOLOv5 up to the latest releases such as
YOLO26/27, with no code changes needed when a new release comes out) on the
three Intel Core Ultra hardware targets of this NUC 15 Pro: **CPU**,
**Arc iGPU**, and **NPU**, via OpenVINO.

## Results on the NUC 15 Pro

Two runs on 2026-09-24, both at 640 px with 10 warm-up + 100 timed
iterations:

- **Sizes n and s** on CPU, GPU and NPU, fp32 + int8 (the default run,
  `python -m yolobench.benchmark`). Raw data:
  [results/summary_default_20260924_val500.csv](results/summary_default_20260924_val500.csv).
- **Sizes m, l and x** on the GPU only, fp32 + int8
  (`--sizes m,l,x --devices GPU`). Raw data:
  [results/summary_mlx_gpu_20260924_val500.csv](results/summary_mlx_gpu_20260924_val500.csv).

mAP50-95 is measured on `coco-val500`, a fixed 500-image subset of COCO
val2017 that the pretrained models never saw during training; INT8 is
calibrated on coco128.

| | |
| --- | --- |
| Machine | ASUS NUC 15 Pro (NUC15CRKU5), Intel Core Ultra 5 225H, 14 GB usable RAM |
| OS | Ubuntu 24.04.5, kernel 7.0.0-34-generic (HWE), power profile `performance` |
| Drivers | compute-runtime 26.35.39758.10, IGC 2.41.5, NPU driver 1.35.0, Level Zero loader 1.28.2 |
| Software | ultralytics 8.4.161, OpenVINO 2026.4.0, NNCF 3.4.0 |

As a sanity check, yolo11n fp32 scores 0.394 here, against 0.395 in the
official Ultralytics figures on the full val2017.

### Best model per FPS budget

The most accurate model on the GPU that still reaches each frame rate. It
is YOLO26 at every budget:

| Needs at least | Model | FPS | Latency p95 | mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| ≥ 80 FPS | **yolo26s int8** | 82.0 | 13.3 ms | 0.478 |
| ≥ 60 FPS | **yolo26m int8** | 62.3 | 16.5 ms | 0.530 |
| ≥ 50 FPS | **yolo26l int8** | 51.5 | 20.2 ms | 0.545 |
| ≥ 30 FPS | **yolo26x int8** | 36.1 | 28.2 ms | 0.570 |

For a 30 FPS camera, each frame must be processed in under 33 ms including
everything else the robot does. yolo26l int8 (p95 20 ms) leaves the most
comfortable margin at high accuracy; yolo26x int8 (p95 28 ms) is tight.

### Highlights

All highlights are measured on the GPU, the fastest device, across every
size. fp32 mAP is the same on CPU, GPU and NPU (within ±0.002), so running
a model elsewhere only costs speed. Each model appears at most once per
category.

#### 🎯 Best mAP

Highest mAP50-95, regardless of speed. yolo26x is well ahead; the other
`x` models are tied around 0.54–0.55:

| Rank | Model | FPS | Latency p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolo26x fp32** | 24.9 | 41.9 ms | **0.581** |
| 🥈 | **yolov10x fp32** | **29.6** | 34.5 ms | 0.547 |
| 🥉 | **yolo11x fp32** | 24.8 | 42.1 ms | 0.545 |

#### ⚡ Best FPS

Highest mean FPS, regardless of accuracy. GPU FPS changed by up to 11%
between two runs of the same model, so these three are effectively a tie:

| Rank | Model | FPS | Latency p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolov10n fp32** | **98.4** | 11.4 ms | 0.401 |
| 🥈 | **yolov5nu int8** | 96.1 | 11.5 ms | 0.344 |
| 🥉 | **yolo26n fp32** | 94.2 | 11.7 ms | **0.413** |

#### ⚖️ Best balance

Highest mAP50-95 among combinations that reach at least 80 FPS, so they
are fast enough for real-time use with plenty of headroom:

| Rank | Model | FPS | Latency p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolo26s int8** | 82.0 | 13.3 ms | **0.478** |
| 🥈 | **yolo11s int8** | 82.0 | 12.9 ms | 0.466 |
| 🥉 | **yolov10s int8** | 81.0 | 13.4 ms | 0.464 |

Key findings:

- **YOLO26 is the best family at every speed**: from yolo26s int8
  (0.478 at 82 FPS) up to yolo26x fp32, the most accurate model tested
  (0.581).
- **yolo26l int8 matches the `x` models of other families** (0.545, like
  yolo11x, yolov8x and yolov10x) while running about 1.3–1.4× faster
  (51.5 FPS vs 36–40 FPS in int8).
- **int8 pays off most on big models**: on the GPU it makes the `x` models
  36–50% faster for at most 0.011 mAP, while the `n` models gain nothing
  (−12% to +1%).
  The exception is **YOLO12**, which gains nothing from int8 at any size and
  is also the slowest family (yolo12x: 14 FPS).
- **The Arc iGPU is the fastest device** for every n/s model: 1.2–3.2× the
  CPU and 1.0–2.2× the NPU.
- **Avoid int8 on the NPU for yolo11n, yolo26n, yolov10n and yolo26s**:
  mAP drops to 0.215, 0.227, 0.246 and 0.426, while the same int8 models
  stay accurate on CPU/GPU. The other n/s models are fine in int8 on the
  NPU; m/l/x were not tested on the NPU.
- All 122 combinations (84 + 38) completed (`ok`), with no crashes.

Caveats: each combination ran once. Compared with an earlier run of the
same n/s models, FPS changed by a median of 4%, by up to 11% on the GPU and
15% on the NPU, and by 37% in one CPU case (yolov9t int8), so small FPS
gaps are noise. The mAP uses 500 COCO images and 80 COCO classes, so gaps
of about ±0.01 are noise too; re-test the top candidates on your own
dataset before choosing a model for a specific task.

### Full results

FPS is the mean over 100 timed iterations; mAP is mAP50-95 on
`coco-val500`. Highlighted rows: 🎯 best mAP, ⚡ best FPS, ⚖️ best
balance.

**Sizes n and s, all devices:**

| Model | Precision | FPS CPU | FPS GPU | FPS NPU | mAP CPU | mAP GPU | mAP NPU |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| yolov5nu | fp32 | 45.8 | 95.2 | 76.1 | 0.347 | 0.346 | 0.345 |
| yolov5nu ⚡ | int8 | 72.9 | 96.1 | 84.8 | 0.345 | 0.344 | 0.344 |
| yolov5su | fp32 | 26.0 | 79.9 | 52.9 | 0.435 | 0.434 | 0.435 |
| yolov5su | int8 | 48.8 | 93.5 | 63.4 | 0.435 | 0.434 | 0.433 |
| yolov8n | fp32 | 43.1 | 92.0 | 73.6 | 0.384 | 0.384 | 0.384 |
| yolov8n | int8 | 75.5 | 91.6 | 78.6 | 0.388 | 0.386 | 0.384 |
| yolov8s | fp32 | 22.6 | 71.4 | 52.0 | 0.446 | 0.445 | 0.444 |
| yolov8s | int8 | 47.9 | 80.4 | 62.9 | 0.444 | 0.443 | 0.442 |
| yolov9t | fp32 | 38.5 | 84.4 | 68.0 | 0.384 | 0.383 | 0.383 |
| yolov9t | int8 | 47.2 | 74.2 | 72.3 | 0.386 | 0.386 | 0.385 |
| yolov9s | fp32 | 21.8 | 63.5 | 49.7 | 0.462 | 0.461 | 0.461 |
| yolov9s | int8 | 37.7 | 66.7 | 58.9 | 0.461 | 0.460 | 0.459 |
| yolov10n ⚡ | fp32 | 52.0 | 98.4 | 72.3 | 0.401 | 0.401 | 0.403 |
| yolov10n | int8 | 67.0 | 92.7 | 55.9 | 0.396 | 0.395 | 0.246 |
| yolov10s | fp32 | 27.0 | 70.9 | 51.7 | 0.470 | 0.469 | 0.469 |
| yolov10s ⚖️ | int8 | 51.9 | 81.0 | 37.6 | 0.466 | 0.464 | 0.449 |
| yolo11n | fp32 | 46.0 | 88.4 | 78.0 | 0.395 | 0.394 | 0.394 |
| yolo11n | int8 | 68.0 | 89.5 | 73.6 | 0.394 | 0.395 | 0.215 |
| yolo11s | fp32 | 26.3 | 69.5 | 51.5 | 0.471 | 0.471 | 0.470 |
| yolo11s ⚖️ | int8 | 46.1 | 82.0 | 56.7 | 0.466 | 0.466 | 0.456 |
| yolo12n | fp32 | 41.2 | 83.0 | 54.4 | 0.409 | 0.409 | 0.409 |
| yolo12n | int8 | 36.0 | 76.3 | 49.9 | 0.405 | 0.406 | 0.406 |
| yolo12s | fp32 | 22.5 | 50.2 | 32.5 | 0.486 | 0.485 | 0.485 |
| yolo12s | int8 | 31.8 | 51.2 | 34.6 | 0.483 | 0.476 | 0.483 |
| yolo26n ⚡ | fp32 | 54.7 | 94.2 | 78.4 | 0.413 | 0.413 | 0.413 |
| yolo26n | int8 | 40.5 | 88.7 | 74.1 | 0.408 | 0.409 | 0.227 |
| yolo26s | fp32 | 27.0 | 71.8 | 54.3 | 0.487 | 0.487 | 0.487 |
| yolo26s ⚖️ | int8 | 45.1 | 82.0 | 55.4 | 0.477 | 0.478 | 0.426 |

**Sizes m, l and x, GPU only:**

| Model | Precision | FPS | Latency p95 | mAP50-95 |
| --- | --- | ---: | ---: | ---: |
| yolov5mu | fp32 | 57.6 | 18.0 ms | 0.485 |
| yolov5mu | int8 | 69.7 | 14.9 ms | 0.481 |
| yolov5lu | fp32 | 43.9 | 23.3 ms | 0.518 |
| yolov5lu | int8 | 58.4 | 17.7 ms | 0.512 |
| yolov5xu | fp32 | 27.0 | 37.5 ms | 0.534 |
| yolov5xu | int8 | 38.2 | 26.6 ms | 0.527 |
| yolov8m | fp32 | 47.8 | 21.3 ms | 0.497 |
| yolov8m | int8 | 60.3 | 17.9 ms | 0.495 |
| yolov8l | fp32 | 35.0 | 30.1 ms | 0.537 |
| yolov8l | int8 | 48.1 | 23.7 ms | 0.525 |
| yolov8x | fp32 | 24.0 | 42.9 ms | 0.542 |
| yolov8x | int8 | 36.1 | 28.8 ms | 0.544 |
| yolov9m | fp32 | 39.9 | 25.5 ms | 0.503 |
| yolov9m | int8 | 40.0 | 25.9 ms | 0.500 |
| yolov10m | fp32 | 51.4 | 20.2 ms | 0.518 |
| yolov10m | int8 | 60.9 | 16.9 ms | 0.511 |
| yolov10l | fp32 | 39.0 | 26.3 ms | 0.535 |
| yolov10l | int8 | 49.3 | 21.7 ms | 0.530 |
| yolov10x 🎯 | fp32 | 29.6 | 34.5 ms | 0.547 |
| yolov10x | int8 | 40.3 | 26.7 ms | 0.543 |
| yolo11m | fp32 | 44.9 | 23.4 ms | 0.509 |
| yolo11m | int8 | 62.7 | 16.8 ms | 0.506 |
| yolo11l | fp32 | 40.6 | 25.5 ms | 0.530 |
| yolo11l | int8 | 50.9 | 20.2 ms | 0.520 |
| yolo11x 🎯 | fp32 | 24.8 | 42.1 ms | 0.545 |
| yolo11x | int8 | 36.1 | 28.2 ms | 0.542 |
| yolo12m | fp32 | 32.3 | 33.3 ms | 0.531 |
| yolo12m | int8 | 32.7 | 34.4 ms | 0.527 |
| yolo12l | fp32 | 22.0 | 46.7 ms | 0.536 |
| yolo12l | int8 | 21.4 | 47.6 ms | 0.533 |
| yolo12x | fp32 | 13.7 | 77.7 ms | 0.544 |
| yolo12x | int8 | 14.2 | 71.3 ms | 0.541 |
| yolo26m | fp32 | 48.3 | 22.0 ms | 0.538 |
| yolo26m | int8 | 62.3 | 16.5 ms | 0.530 |
| yolo26l | fp32 | 41.5 | 24.8 ms | 0.542 |
| yolo26l | int8 | 51.5 | 20.2 ms | 0.545 |
| yolo26x 🎯 | fp32 | 24.9 | 41.9 ms | 0.581 |
| yolo26x | int8 | 36.1 | 28.2 ms | 0.570 |

## 1. Prerequisites

- Windows 11 on an Asus NUC 15 Pro (Intel Core Ultra chip).
- Admin rights to install the Intel Arc (iGPU) and Intel NPU drivers. These
  are **separate** drivers, so don't just install the standard graphics
  driver and assume the NPU comes with it. Search for "Intel NPU Driver" on
  Intel's download page for the matching chip.
- Running on **Ubuntu 24.04** instead? The Python code is the same; only the
  setup and drivers differ. See [section 8](#8-running-on-ubuntu-2404).

## 2. Install Python

**Python 3.10–3.12** is recommended. Avoid Python 3.13+ for now, because
`openvino`/`nncf` wheels on PyPI are sometimes not yet available for the
newest Python release. Check the installed version:

```powershell
python --version
```

If it's missing, download it from <https://www.python.org/downloads/>
(3.12 recommended) and make sure "Add python.exe to PATH" is checked during
installation.

## 3. Set up the environment

From the project root, run:

```powershell
scripts\setup_env.ps1
```

This script creates a virtual environment (`.venv`), installs every
dependency in `requirements.txt`, and then checks which OpenVINO devices are
detected on this NUC.

Manual setup (if you want full control):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. Verify devices

`setup_env.ps1` already runs this automatically, but you can re-check at any
time:

```powershell
.venv\Scripts\python.exe -c "from openvino import Core; print(Core().available_devices)"
```

- At minimum you should see `['CPU']`.
- With the latest Intel Arc driver installed: `['CPU', 'GPU']`.
- With the NPU driver installed: `['CPU', 'GPU', 'NPU']`.

If `GPU` or `NPU` doesn't show up, the benchmark still runs normally: an
undetected device is skipped automatically (not an error), but of course
there will be no results for that device until its driver is installed.

## 5. Run the benchmark

Quick run (default: nano+small sizes, fp32+int8 precision, every detected
device):

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark
```

More specific runs:

```powershell
# Only YOLO11 and YOLO26, size n only, all precisions, CPU & NPU only
.venv\Scripts\python.exe -m yolobench.benchmark --families yolo11,yolo26 --sizes n --precisions fp32,fp16,int8 --devices CPU,NPU

# Full sweep (every model x every size x every precision x every device)
.venv\Scripts\python.exe -m yolobench.benchmark --sizes n,s,m,l,x --precisions fp32,fp16,int8 --devices CPU,GPU,NPU
```

A full sweep can take a long time (hours, depending on the number of
models). It's safe to leave running, because every combination (model ×
precision × device) is written to `results/raw/<timestamp>.json` as soon as
it finishes, so if the run stops midway (Ctrl+C, power loss, etc.), finished
results are not lost.

To prepare every model up front (download weights + dataset and export all
precisions, without benchmarking), add `--prepare`. The resulting `models/`
and `data/` folders can be copied to another machine and benchmarked offline
(see [TEST_STEP.md](TEST_STEP.md) step 2b, in Indonesian).

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark --prepare --sizes n,s,m,l,x --precisions fp32,fp16,int8
```

Datasets: INT8 calibration uses `coco128.yaml` (`--calib-dataset`) and mAP
is measured on `coco-val500` (`--val-dataset`), a fixed 500-image subset of
COCO val2017 that is downloaded automatically (~1 GB) the first time.
`coco-val<N>` picks another subset size, and a path to your own dataset yaml
works for either option.

Every option can also be set permanently in `configs/benchmark.yaml` so you
don't have to type long flags each time.

## 6. Reading the results

- **`results/benchmark_summary.csv`**: one row per combination
  (model × precision × device), with `fps_mean`, `latency_ms_mean`,
  `latency_ms_p95`, `map50_95`, `map50`, `val_dataset` (the dataset the mAP
  was measured on), `model_size_mb`, `status`, and `error` (on failure).
- At the end of a run, the terminal also prints pivot summary tables:
  rows = model, columns = device, values = mean FPS (and a second table for
  mAP).
- `results/raw/<timestamp>.json`: raw per-run data for auditing/history.

## 7. Important notes

- The OpenVINO **NPU** generally needs **INT8** models for best performance,
  and supports fewer operators than CPU/GPU. If a model+device combination
  fails on the NPU, it's recorded as an `inference_failed` row in the CSV
  (instead of crashing the whole benchmark); see the `error` column for
  details. Every combination runs in its own process, so even a native crash
  (segfault) in the GPU/NPU plugin is only recorded as a `crashed` row and
  the benchmark moves on to the next combination.
- **File locations**: `.pt` weights are downloaded to `models/weights/`,
  OpenVINO exports go to `models/<family>/<size>/<precision>_openvino_model/`,
  the datasets to `data/` (coco128 for calibration, COCO val2017 under
  `data/coco/`), and validation output to `results/runs/`.
  This project uses its own ultralytics settings
  (`data/ultralytics_config/`), so your global ultralytics settings are left
  untouched.
- **Benchmarking on a laptop/other machine**: plug it in and use a
  performance power mode, because on battery the CPU/GPU is throttled and
  FPS is much lower.
- **Calibration and validation data are kept separate**: INT8 is
  calibrated on coco128 (train2017 images), while mAP is measured on COCO
  val2017, which the pretrained models never saw during training. Don't
  validate on `coco128.yaml`: its `val:` split is the same train2017 images,
  so it rewards memorisation and makes INT8 look better than it is.
- **mAP here is close to, but not exactly, the official figures**: 500 of
  the 5,000 val2017 images are used to keep runs fast, so expect small
  differences from the mAP in papers/model cards (about ±0.01). Use the
  numbers mainly for **comparison** between models, devices and precisions
  on this machine.
- According to the Ultralytics docs, OpenVINO NPU support requires an
  **Intel Core Ultra Series 2xxV / 3xx or newer** chip. The NUC 15 Pro most
  likely meets this, but still verify it with step 4 above, since it hasn't
  been confirmed here that the NPU driver ships preinstalled on this NUC's
  Windows image.

## 8. Running on Ubuntu 24.04

The benchmark code itself has nothing Windows-specific, so it runs unchanged
on Ubuntu 24.04. Only the setup steps and drivers differ.

**Drivers** (the CPU works without any of these):

- **Kernel**: install the HWE kernel, because the stock 6.8 kernel of 24.04
  may be too old for the Core Ultra 200-series iGPU/NPU:
  `sudo apt install linux-generic-hwe-24.04`, then reboot.
- **Arc iGPU**: install Intel's compute runtime (OpenCL/Level Zero), either
  `sudo apt install intel-opencl-icd` or the latest `.deb` packages from
  <https://github.com/intel/compute-runtime/releases>.
- **NPU**: install the `.deb` packages for Ubuntu 24.04 from
  <https://github.com/intel/linux-npu-driver/releases> (the kernel module
  `intel_vpu` is already part of the kernel).
- **Permissions**: add your user to the `render` group, otherwise GPU/NPU
  are often not detected: `sudo usermod -aG render $USER`, then log out and
  back in.

**Setup** (Python 3.12 ships with 24.04, which is within the recommended
range):

```bash
sudo apt install python3-venv
bash scripts/setup_env.sh
```

`setup_env.sh` is the Linux counterpart of `setup_env.ps1`: it creates
`.venv`, installs the dependencies, warns if you are not in the `render`
group, and prints the detected OpenVINO devices. To use a different
interpreter, run `PYTHON=python3.12 bash scripts/setup_env.sh`.

**Running**: every command in this README works the same; just replace
`.venv\Scripts\python.exe` with `.venv/bin/python`, for example:

```bash
.venv/bin/python -m yolobench.benchmark
.venv/bin/python -c "from openvino import Core; print(Core().available_devices)"
```

## 9. Live camera test

The benchmark times one model on one still image on an idle machine. To
check a real setup (several cameras at once, tracking, the CPU busy with
other work), run the camera test on already exported models:

```bash
# Interactive: pick camera, model, precision and device from menus
.venv/bin/python -m yolobench.camera

# Models already exported in models/
.venv/bin/python -m yolobench.camera --list

# One camera on the GPU
.venv/bin/python -m yolobench.camera --stream 0:yolo26s:int8:GPU

# Two cameras, one on the GPU and one on the NPU, with tracking, for 10 minutes
.venv/bin/python -m yolobench.camera --stream 0:yolo26s:int8:GPU --stream 2:yolo26s:fp32:NPU --track --duration 600
```

- A stream is `SOURCE:MODEL:PRECISION:DEVICE`. `SOURCE` is a camera index,
  a video file (played back at its own frame rate, like a camera) or a
  stream URL such as `rtsp://...`.
- All streams run at the same time. Each always processes the newest frame,
  so a model that is too slow skips frames instead of falling behind.
- Every 5 s the terminal shows FPS and p95 inference time per stream, plus
  CPU load and temperature. At the end it prints, per stream: processed vs
  camera FPS, skipped frames, inference mean/p95/p99, frame-to-result
  p95/p99, and **OK** or **TOO SLOW** (more than 5% of frames skipped, or
  p95 inference longer than one camera frame, 33 ms at 30 FPS). The summary
  is also saved to `results/camera/<timestamp>.csv`.
- Press `q` in a preview window or Ctrl+C to stop. Use `--no-show` over SSH,
  `--cam-fps`/`--width`/`--height`/`--fourcc` to request a camera mode.
- `--track` uses Ultralytics' ByteTrack, which needs the `lap` package;
  Ultralytics installs it automatically the first time if it's missing.
- For a realistic check, run it for 10-30 minutes with the rest of the
  robot software (ROS, planning, ...) running, and watch p99 and the CPU
  temperature. The step-by-step procedure is in
  [TEST_CAMERA.md](TEST_CAMERA.md) (in Indonesian).
