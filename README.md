# YOLO Benchmark — Asus NUC 15 Pro

Benchmark semua versi model YOLO yang didukung package `ultralytics`
(otomatis terdeteksi — dari YOLOv5 sampai versi terbaru seperti YOLO26/27,
tanpa perlu update kode saat rilis baru muncul) di tiga target hardware
Intel Core Ultra pada NUC 15 Pro ini: **CPU**, **iGPU Arc**, dan **NPU**,
lewat OpenVINO.

## 1. Prasyarat

- Windows 11 di Asus NUC 15 Pro (chip Intel Core Ultra).
- Hak admin untuk instal driver Intel Arc (iGPU) dan Intel NPU — dua driver
  ini **terpisah**, jangan cuma instal driver grafis standar dan
  mengasumsikan NPU otomatis ikut. Cari "Intel NPU Driver" di halaman
  download Intel untuk chip yang sesuai.

## 2. Instal Python

Direkomendasikan **Python 3.10–3.12**. Hindari Python 3.13+ untuk saat ini —
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

Kalau `GPU` atau `NPU` tidak muncul, benchmark tetap jalan normal — device
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

# Full sweep (semua model x ukuran n/s x semua presisi x semua device)
.venv\Scripts\python.exe -m yolobench.benchmark --sizes n,s,m,l,x --precisions fp32,fp16,int8 --devices CPU,GPU,NPU
```

Full sweep bisa berjalan lama (bisa berjam-jam tergantung jumlah model).
Aman ditinggal — setiap kombinasi (model × presisi × device) langsung
ditulis ke `results/raw/<timestamp>.json` begitu selesai, jadi kalau
terhenti di tengah jalan (Ctrl+C, listrik mati, dll), hasil yang sudah
selesai tidak hilang.

Semua opsi bisa juga diatur permanen lewat `configs/benchmark.yaml` supaya
tidak perlu ketik flag panjang tiap kali.

## 6. Membaca hasil

- **`results/benchmark_summary.csv`** — satu baris per kombinasi
  (model × presisi × device), berisi `fps_mean`, `latency_ms_mean`,
  `latency_ms_p95`, `map50_95`, `map50`, `model_size_mb`, `status`, dan
  `error` (kalau gagal).
- Di akhir run, terminal juga menampilkan tabel ringkasan pivot: baris =
  model, kolom = device, isi = FPS rata-rata (dan tabel kedua untuk mAP).
- `results/raw/<timestamp>.json` — data mentah per run untuk audit/histori.

## 7. Catatan penting

- **NPU** OpenVINO umumnya butuh model **INT8** untuk performa terbaik, dan
  punya dukungan operator yang lebih terbatas dibanding CPU/GPU — kalau
  suatu kombinasi model+device gagal di NPU, itu tercatat sebagai baris
  `inference_failed` di CSV (bukan meng-crash seluruh benchmark), lihat
  kolom `error` untuk detail. Setiap kombinasi dijalankan di proses
  terpisah, jadi crash native (segfault) di plugin GPU/NPU pun hanya
  tercatat sebagai baris `crashed` dan benchmark lanjut ke kombinasi
  berikutnya.
- **Lokasi file**: bobot `.pt` diunduh ke `models/weights/`, hasil export
  OpenVINO ke `models/<family>/<size>/<presisi>_openvino_model/`, dataset
  coco128 ke `data/`, dan output validasi ke `results/runs/`. Project ini
  memakai setting ultralytics sendiri (`data/ultralytics_config/`), jadi
  setting ultralytics global di komputermu tidak diubah.
- **Benchmark di laptop/mesin lain**: colok ke listrik dan pakai power mode
  performa — di mode baterai CPU/GPU di-throttle dan FPS jauh lebih rendah.
- **mAP di sini indikatif, bukan angka paper-comparable** — kalibrasi INT8
  dan validasi memakai subset COCO kecil (`coco128.yaml`) untuk mempercepat
  run, jadi gunakan angkanya untuk **perbandingan relatif** antar
  device/presisi pada mesin ini, bukan untuk dibandingkan ke angka mAP resmi
  di paper/model card.
- Dukungan NPU OpenVINO menurut dokumentasi Ultralytics mensyaratkan chip
  **Intel Core Ultra Series 2xxV / 3xx ke atas** — NUC 15 Pro kemungkinan
  besar memenuhi ini, tapi tetap verifikasi lewat langkah 4 di atas karena
  belum dipastikan dari sini secara pasti driver NPU sudah terpasang
  default di image Windows NUC ini.
