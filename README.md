# YOLO Benchmark: Asus NUC 15 Pro

**English** | [Bahasa Indonesia](README.id.md)

Benchmarks every YOLO model version supported by the `ultralytics` package
(detected automatically, from YOLOv5 up to the latest releases such as
YOLO26/27, with no code changes needed when a new release comes out) on the
three Intel Core Ultra hardware targets of this NUC 15 Pro: **CPU**,
**Arc iGPU**, and **NPU**, via OpenVINO.

## 1. Prerequisites

- Windows 11 on an Asus NUC 15 Pro (Intel Core Ultra chip).
- Admin rights to install the Intel Arc (iGPU) and Intel NPU drivers. These
  are **separate** drivers, so don't just install the standard graphics
  driver and assume the NPU comes with it. Search for "Intel NPU Driver" on
  Intel's download page for the matching chip.

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

Every option can also be set permanently in `configs/benchmark.yaml` so you
don't have to type long flags each time.

## 6. Reading the results

- **`results/benchmark_summary.csv`**: one row per combination
  (model × precision × device), with `fps_mean`, `latency_ms_mean`,
  `latency_ms_p95`, `map50_95`, `map50`, `model_size_mb`, `status`, and
  `error` (on failure).
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
  the coco128 dataset to `data/`, and validation output to `results/runs/`.
  This project uses its own ultralytics settings
  (`data/ultralytics_config/`), so your global ultralytics settings are left
  untouched.
- **Benchmarking on a laptop/other machine**: plug it in and use a
  performance power mode, because on battery the CPU/GPU is throttled and
  FPS is much lower.
- **mAP here is indicative, not paper-comparable**: INT8 calibration and
  validation use a small COCO subset (`coco128.yaml`) to keep runs fast, so
  use the numbers for **relative comparison** between devices/precisions on
  this machine, not against official mAP figures in papers/model cards.
- According to the Ultralytics docs, OpenVINO NPU support requires an
  **Intel Core Ultra Series 2xxV / 3xx or newer** chip. The NUC 15 Pro most
  likely meets this, but still verify it with step 4 above, since it hasn't
  been confirmed here that the NPU driver ships preinstalled on this NUC's
  Windows image.
