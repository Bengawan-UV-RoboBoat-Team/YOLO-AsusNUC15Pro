# YOLO Benchmark: Asus NUC 15 Pro

**English** | [Bahasa Indonesia](README.id.md)

Benchmarks every YOLO model version supported by the `ultralytics` package
(detected automatically, from YOLOv5 up to the latest releases such as
YOLO26/27, with no code changes needed when a new release comes out) on the
three Intel Core Ultra hardware targets of this NUC 15 Pro: **CPU**,
**Arc iGPU**, and **NPU**, via OpenVINO.

## Results on the NUC 15 Pro

Default run (`python -m yolobench.benchmark`: sizes n+s, fp32+int8,
CPU/GPU/NPU, 640 px, 10 warm-up + 100 timed iterations) on 2026-09-24.
mAP50-95 is measured on `coco-val500`, a fixed 500-image subset of COCO
val2017 that the pretrained models never saw during training; INT8 is
calibrated on coco128. Raw data:
[results/summary_default_20260924_val500.csv](results/summary_default_20260924_val500.csv).

| | |
| --- | --- |
| Machine | ASUS NUC 15 Pro (NUC15CRKU5), Intel Core Ultra 5 225H, 14 GB usable RAM |
| OS | Ubuntu 24.04.5, kernel 7.0.0-34-generic (HWE), power profile `performance` |
| Drivers | compute-runtime 26.35.39758.10, IGC 2.41.5, NPU driver 1.35.0, Level Zero loader 1.28.2 |
| Software | ultralytics 8.4.161, OpenVINO 2026.4.0, NNCF 3.4.0 |

As a sanity check, yolo11n fp32 scores 0.394 here, against 0.395 in the
official Ultralytics figures on the full val2017.

### Highlights

All highlights are measured on the GPU, the fastest device. fp32 mAP is
the same on CPU, GPU and NPU (within ±0.002), so running a model elsewhere
only costs speed. Each model appears at most once per category.

#### 🎯 Best mAP

Highest mAP50-95, regardless of speed. yolo26s and yolo12s are tied on
accuracy, but yolo26s is about 40% faster:

| Rank | Model | FPS | Latency p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolo26s fp32** | **71.8** | 14.6 ms | **0.487** |
| 🥈 | **yolo12s fp32** | 50.2 | 20.9 ms | 0.485 |
| 🥉 | **yolo11s fp32** | 69.5 | 15.5 ms | 0.471 |

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
are fast enough for real-time use:

| Rank | Model | FPS | Latency p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolo26s int8** | 82.0 | 13.3 ms | **0.478** |
| 🥈 | **yolo11s int8** | 82.0 | 12.9 ms | 0.466 |
| 🥉 | **yolov10s int8** | 81.0 | 13.4 ms | 0.464 |

Key findings:

- **yolo26s is the best all-round choice**: the most accurate model
  (0.487 in fp32), and in int8 it keeps 0.478 at 82 FPS on the GPU.
- **The Arc iGPU is the fastest device** for every model: 1.2–3.2× the
  CPU and 1.0–2.2× the NPU.
- **`s` + int8 on the GPU is the sweet spot.** Compared with `n` int8,
  it costs only about 8% FPS on the GPU but gains about +0.07 mAP
  (yolo26: 88.7 → 82.0 FPS, 0.409 → 0.478).
- **Avoid int8 on the NPU for yolo11n, yolo26n, yolov10n and yolo26s**:
  mAP drops to 0.215, 0.227, 0.246 and 0.426, while the same int8 models
  stay accurate on CPU/GPU. The other models are fine in int8 on the NPU.
- All 84 combinations completed (`ok`), with no crashes.

Caveats: this is a single run. Compared with an earlier run of the same
models, FPS changed by a median of 4%, by up to 11% on the GPU and 15% on
the NPU, and by 37% in one CPU case (yolov9t int8), so small FPS gaps are
noise. The mAP uses 500 COCO images and 80 COCO classes; re-test the top
candidates on your own dataset before choosing a model for a specific task.

### Full results

FPS is the mean over 100 timed iterations; mAP is mAP50-95 on
`coco-val500`. Highlighted rows: 🎯 best mAP, ⚡ best FPS, ⚖️ best
balance.

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
| yolo11s 🎯 | fp32 | 26.3 | 69.5 | 51.5 | 0.471 | 0.471 | 0.470 |
| yolo11s ⚖️ | int8 | 46.1 | 82.0 | 56.7 | 0.466 | 0.466 | 0.456 |
| yolo12n | fp32 | 41.2 | 83.0 | 54.4 | 0.409 | 0.409 | 0.409 |
| yolo12n | int8 | 36.0 | 76.3 | 49.9 | 0.405 | 0.406 | 0.406 |
| yolo12s 🎯 | fp32 | 22.5 | 50.2 | 32.5 | 0.486 | 0.485 | 0.485 |
| yolo12s | int8 | 31.8 | 51.2 | 34.6 | 0.483 | 0.476 | 0.483 |
| yolo26n ⚡ | fp32 | 54.7 | 94.2 | 78.4 | 0.413 | 0.413 | 0.413 |
| yolo26n | int8 | 40.5 | 88.7 | 74.1 | 0.408 | 0.409 | 0.227 |
| yolo26s 🎯 | fp32 | 27.0 | 71.8 | 54.3 | 0.487 | 0.487 | 0.487 |
| yolo26s ⚖️ | int8 | 45.1 | 82.0 | 55.4 | 0.477 | 0.478 | 0.426 |


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
