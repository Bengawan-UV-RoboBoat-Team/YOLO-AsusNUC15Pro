# YOLO Benchmark: Asus NUC 15 Pro

[English](README.md) | **Bahasa Indonesia**

Benchmark semua versi model YOLO yang didukung package `ultralytics`
(otomatis terdeteksi, dari YOLOv5 sampai versi terbaru seperti YOLO26/27,
tanpa perlu update kode saat rilis baru muncul) di tiga target hardware
Intel Core Ultra pada NUC 15 Pro ini: **CPU**, **iGPU Arc**, dan **NPU**,
lewat OpenVINO.

## Hasil di NUC 15 Pro

Run default (`python -m yolobench.benchmark`: ukuran n+s, fp32+int8,
CPU/GPU/NPU, 640 px, 10 iterasi warm-up + 100 iterasi terukur) pada
2026-09-24. mAP50-95 diukur di `coco-val500`, yaitu subset tetap 500 gambar
COCO val2017 yang tidak pernah dilihat model pretrained saat training.
INT8 dikalibrasi di coco128. Data mentah:
[results/summary_default_20260924_val500.csv](results/summary_default_20260924_val500.csv).

| | |
| --- | --- |
| Mesin | ASUS NUC 15 Pro (NUC15CRKU5), Intel Core Ultra 5 225H, RAM terpakai 14 GB |
| OS | Ubuntu 24.04.5, kernel 7.0.0-34-generic (HWE), power profile `performance` |
| Driver | compute-runtime 26.35.39758.10, IGC 2.41.5, driver NPU 1.35.0, Level Zero loader 1.28.2 |
| Software | ultralytics 8.4.161, OpenVINO 2026.4.0, NNCF 3.4.0 |

Sebagai cek kewajaran, yolo11n fp32 di sini mendapat 0.394, sedangkan
angka resmi Ultralytics di val2017 penuh adalah 0.395.

### Highlight

Semua highlight diukur di GPU, device tercepat. mAP fp32 sama di CPU, GPU,
dan NPU (selisih maksimal ±0.002), jadi menjalankan model di device lain
hanya mengorbankan kecepatan. Satu model hanya muncul sekali per kategori.

#### 🎯 mAP terbaik

mAP50-95 tertinggi, tanpa melihat kecepatan. yolo26s dan yolo12s seri
dalam akurasi, tapi yolo26s sekitar 40% lebih cepat:

| Peringkat | Model | FPS | Latensi p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolo26s fp32** | **71.8** | 14.6 ms | **0.487** |
| 🥈 | **yolo12s fp32** | 50.2 | 20.9 ms | 0.485 |
| 🥉 | **yolo11s fp32** | 69.5 | 15.5 ms | 0.471 |

#### ⚡ FPS terbaik

FPS rata-rata tertinggi, tanpa melihat akurasi. FPS di GPU berubah sampai
11% antara dua run model yang sama, jadi ketiganya praktis seri:

| Peringkat | Model | FPS | Latensi p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolov10n fp32** | **98.4** | 11.4 ms | 0.401 |
| 🥈 | **yolov5nu int8** | 96.1 | 11.5 ms | 0.344 |
| 🥉 | **yolo26n fp32** | 94.2 | 11.7 ms | **0.413** |

#### ⚖️ Paling seimbang

mAP50-95 tertinggi di antara kombinasi yang mencapai minimal 80 FPS,
sehingga cukup cepat untuk real-time:

| Peringkat | Model | FPS | Latensi p95 | mAP50-95 |
| :---: | --- | ---: | ---: | ---: |
| 🥇 | **yolo26s int8** | 82.0 | 13.3 ms | **0.478** |
| 🥈 | **yolo11s int8** | 82.0 | 12.9 ms | 0.466 |
| 🥉 | **yolov10s int8** | 81.0 | 13.4 ms | 0.464 |

Temuan utama:

- **yolo26s adalah pilihan terbaik secara keseluruhan**: model paling
  akurat (0.487 di fp32), dan di int8 tetap 0.478 pada 82 FPS di GPU.
- **iGPU Arc adalah device tercepat** untuk semua model: 1.2–3.2× CPU dan
  1.0–2.2× NPU.
- **`s` + int8 di GPU adalah pilihan paling efisien.** Dibanding `n` int8,
  FPS-nya hanya turun sekitar 8% di GPU, tapi mAP naik sekitar +0.07
  (yolo26: 88.7 → 82.0 FPS, 0.409 → 0.478).
- **Hindari int8 di NPU untuk yolo11n, yolo26n, yolov10n, dan yolo26s**:
  mAP-nya turun ke 0.215, 0.227, 0.246, dan 0.426, padahal model int8 yang
  sama tetap akurat di CPU/GPU. Model lain aman memakai int8 di NPU.
- Ke-84 kombinasi selesai (`ok`), tanpa crash.

Catatan: ini hasil satu kali run. Dibanding run sebelumnya untuk model yang
sama, FPS berubah dengan median 4%, sampai 11% di GPU dan 15% di NPU, serta
37% pada satu kasus CPU (yolov9t int8). Jadi selisih FPS yang kecil adalah
noise. mAP memakai 500 gambar dan 80 kelas COCO; uji ulang kandidat teratas
di dataset sendiri sebelum memilih model untuk tugas tertentu.

### Hasil lengkap

FPS adalah rata-rata dari 100 iterasi terukur; mAP adalah mAP50-95 di
`coco-val500`. Baris highlight: 🎯 mAP terbaik, ⚡ FPS terbaik, ⚖️ paling
seimbang.

| Model | Presisi | FPS CPU | FPS GPU | FPS NPU | mAP CPU | mAP GPU | mAP NPU |
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

## 1. Prasyarat

- Windows 11 di Asus NUC 15 Pro (chip Intel Core Ultra).
- Hak admin untuk instal driver Intel Arc (iGPU) dan Intel NPU. Dua driver
  ini **terpisah**, jadi jangan cuma instal driver grafis standar dan
  mengasumsikan NPU otomatis ikut. Cari "Intel NPU Driver" di halaman
  download Intel untuk chip yang sesuai.
- Mau jalan di **Ubuntu 24.04**? Kode Python-nya sama, yang beda cuma setup
  dan driver. Lihat [bagian 8](#8-menjalankan-di-ubuntu-2404).

## 2. Instal Python

Direkomendasikan **Python 3.10–3.12**. Hindari Python 3.13+ untuk saat ini karena
wheel `openvino`/`nncf` di PyPI kadang belum tersedia untuk versi Python
paling baru. Cek versi yang terinstall:

```powershell
python --version
```

Kalau belum ada, unduh dari https://www.python.org/downloads/ (versi 3.12
direkomendasikan), lalu pastikan dicentang "Add python.exe to PATH" saat
instalasi.

## 3. Setup environment

Dari root project ini, jalankan:

```powershell
scripts\setup_env.ps1
```

Script ini akan: membuat virtual environment (`.venv`), install semua
dependency di `requirements.txt`, lalu langsung cek device OpenVINO yang
terdeteksi di NUC ini.

Setup manual (kalau mau kontrol sendiri):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

## 4. Verifikasi device

`setup_env.ps1` sudah otomatis menjalankan ini, tapi bisa dicek ulang kapan
saja:

```powershell
.venv\Scripts\python.exe -c "from openvino import Core; print(Core().available_devices)"
```

- Minimal harus muncul `['CPU']`.
- Setelah driver Intel Arc terinstall dan terbaru: `['CPU', 'GPU']`.
- Setelah driver NPU terinstall: `['CPU', 'GPU', 'NPU']`.

Kalau `GPU` atau `NPU` tidak muncul, benchmark tetap jalan normal: device
yang tidak terdeteksi otomatis di-skip (bukan error), tapi tentu saja tidak
akan ada hasil untuk device tersebut sampai drivernya diinstal.

## 5. Jalankan benchmark

Run cepat (default: ukuran nano+small, presisi fp32+int8, semua device yang
terdeteksi):

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark
```

Contoh run yang lebih spesifik:

```powershell
# Hanya YOLO11 dan YOLO26, ukuran n saja, semua presisi, hanya CPU & NPU
.venv\Scripts\python.exe -m yolobench.benchmark --families yolo11,yolo26 --sizes n --precisions fp32,fp16,int8 --devices CPU,NPU

# Full sweep (semua model x semua ukuran x semua presisi x semua device)
.venv\Scripts\python.exe -m yolobench.benchmark --sizes n,s,m,l,x --precisions fp32,fp16,int8 --devices CPU,GPU,NPU
```

Full sweep bisa berjalan lama (bisa berjam-jam tergantung jumlah model).
Aman ditinggal karena setiap kombinasi (model × presisi × device) langsung
ditulis ke `results/raw/<timestamp>.json` begitu selesai, jadi kalau
terhenti di tengah jalan (Ctrl+C, listrik mati, dll), hasil yang sudah
selesai tidak hilang.

Untuk menyiapkan semua model di awal (download bobot + dataset dan export
semua presisi, tanpa benchmark), tambahkan `--prepare`. Hasilnya di
`models/` dan `data/` bisa di-copy ke mesin lain lalu dibenchmark offline
(lihat [TEST_STEP.md](TEST_STEP.md) Tahap 2b).

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark --prepare --sizes n,s,m,l,x --precisions fp32,fp16,int8
```

Dataset: kalibrasi INT8 memakai `coco128.yaml` (`--calib-dataset`), dan
mAP diukur di `coco-val500` (`--val-dataset`), yaitu subset tetap 500
gambar dari COCO val2017 yang diunduh otomatis (±1 GB) saat pertama kali
dipakai. `coco-val<N>` memilih ukuran subset lain, dan path ke yaml
dataset sendiri bisa dipakai untuk kedua opsi.

Semua opsi bisa juga diatur permanen lewat `configs/benchmark.yaml` supaya
tidak perlu ketik flag panjang tiap kali.

## 6. Membaca hasil

- **`results/benchmark_summary.csv`**: satu baris per kombinasi
  (model × presisi × device), berisi `fps_mean`, `latency_ms_mean`,
  `latency_ms_p95`, `map50_95`, `map50`, `val_dataset` (dataset tempat mAP
  diukur), `model_size_mb`, `status`, dan `error` (kalau gagal).
- Di akhir run, terminal juga menampilkan tabel ringkasan pivot: baris =
  model, kolom = device, isi = FPS rata-rata (dan tabel kedua untuk mAP).
- `results/raw/<timestamp>.json`: data mentah per run untuk audit/histori.

## 7. Catatan penting

- **NPU** OpenVINO umumnya butuh model **INT8** untuk performa terbaik, dan
  punya dukungan operator yang lebih terbatas dibanding CPU/GPU. Kalau
  suatu kombinasi model+device gagal di NPU, itu tercatat sebagai baris
  `inference_failed` di CSV (bukan meng-crash seluruh benchmark), lihat
  kolom `error` untuk detail. Setiap kombinasi dijalankan di proses
  terpisah, jadi crash native (segfault) di plugin GPU/NPU pun hanya
  tercatat sebagai baris `crashed` dan benchmark lanjut ke kombinasi
  berikutnya.
- **Lokasi file**: bobot `.pt` diunduh ke `models/weights/`, hasil export
  OpenVINO ke `models/<family>/<size>/<presisi>_openvino_model/`, dataset
  ke `data/` (coco128 untuk kalibrasi, COCO val2017 di `data/coco/`), dan
  output validasi ke `results/runs/`. Project ini
  memakai setting ultralytics sendiri (`data/ultralytics_config/`), jadi
  setting ultralytics global di komputermu tidak diubah.
- **Benchmark di laptop/mesin lain**: colok ke listrik dan pakai power mode
  performa, karena di mode baterai CPU/GPU di-throttle dan FPS jauh lebih rendah.
- **Data kalibrasi dan validasi dipisah**: INT8 dikalibrasi di coco128
  (gambar train2017), sedangkan mAP diukur di COCO val2017 yang tidak
  pernah dilihat model pretrained saat training. Jangan validasi di
  `coco128.yaml`: split `val:`-nya adalah gambar train2017 yang sama, jadi
  yang terukur adalah hafalan model dan INT8 terlihat lebih baik dari
  kenyataannya.
- **mAP di sini mendekati, tapi tidak persis sama dengan, angka resmi**:
  hanya 500 dari 5.000 gambar val2017 yang dipakai supaya run cepat, jadi
  wajar ada selisih kecil (±0,01) dari mAP di paper/model card. Gunakan
  angkanya terutama untuk **perbandingan** antar model, device, dan presisi
  di mesin ini.
- Dukungan NPU OpenVINO menurut dokumentasi Ultralytics mensyaratkan chip
  **Intel Core Ultra Series 2xxV / 3xx ke atas**: NUC 15 Pro kemungkinan
  besar memenuhi ini, tapi tetap verifikasi lewat langkah 4 di atas karena
  belum dipastikan dari sini secara pasti driver NPU sudah terpasang
  default di image Windows NUC ini.

## 8. Menjalankan di Ubuntu 24.04

Kode benchmark-nya sendiri tidak punya bagian khusus Windows, jadi jalan
tanpa perubahan di Ubuntu 24.04. Yang beda cuma langkah setup dan driver.

**Driver** (CPU tetap jalan tanpa semua ini):

- **Kernel**: pasang kernel HWE, karena kernel 6.8 bawaan 24.04 bisa jadi
  terlalu lama untuk iGPU/NPU Core Ultra seri 200:
  `sudo apt install linux-generic-hwe-24.04`, lalu reboot.
- **iGPU Arc**: pasang compute runtime Intel (OpenCL/Level Zero), lewat
  `sudo apt install intel-opencl-icd` atau paket `.deb` terbaru dari
  <https://github.com/intel/compute-runtime/releases>.
- **NPU**: pasang paket `.deb` untuk Ubuntu 24.04 dari
  <https://github.com/intel/linux-npu-driver/releases> (modul kernel
  `intel_vpu` sudah ada di kernel).
- **Permission**: masukkan user ke grup `render`, kalau tidak GPU/NPU
  sering tidak terdeteksi: `sudo usermod -aG render $USER`, lalu logout
  dan login lagi.

**Setup** (Python bawaan 24.04 adalah 3.12, masih dalam rentang yang
direkomendasikan):

```bash
sudo apt install python3-venv
bash scripts/setup_env.sh
```

`setup_env.sh` adalah versi Linux dari `setup_env.ps1`: membuat `.venv`,
install dependency, memberi peringatan kalau user belum masuk grup
`render`, lalu menampilkan device OpenVINO yang terdeteksi. Untuk memakai
interpreter lain, jalankan `PYTHON=python3.12 bash scripts/setup_env.sh`.

**Menjalankan**: semua perintah di README ini sama saja, cukup ganti
`.venv\Scripts\python.exe` dengan `.venv/bin/python`, contohnya:

```bash
.venv/bin/python -m yolobench.benchmark
.venv/bin/python -c "from openvino import Core; print(Core().available_devices)"
```
