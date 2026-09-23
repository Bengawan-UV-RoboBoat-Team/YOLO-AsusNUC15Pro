# Langkah Pengujian — Asus NUC 15 Pro

Panduan urut untuk menjalankan benchmark YOLO di NUC 15 Pro, dari mesin
kosong sampai hasil tersimpan. Kerjakan berurutan; tiap langkah punya
**cek lolos** — jangan lanjut sebelum cek itu terpenuhi.

Semua perintah dijalankan di **PowerShell** dari root folder project.

---

## Tahap 0 — Persiapan mesin

1. **Colok NUC ke listrik** dan set power mode ke performa maksimal:
   Settings → System → Power → Power mode → **Best performance**.
   Mode hemat daya men-throttle CPU/GPU dan membuat FPS jauh lebih rendah.
2. **Tutup aplikasi berat lain** (browser dengan banyak tab, game, sync
   OneDrive besar) supaya tidak mengganggu pengukuran.
3. **Update Windows** dan restart sekali sebelum mulai.

**Cek lolos:** NUC tersambung listrik, power mode = Best performance.

---

## Tahap 1 — Driver

Dua driver ini **terpisah**, keduanya harus diinstal:

1. **Intel Arc Graphics driver** (untuk iGPU) — dari halaman download Intel,
   pilih driver terbaru untuk Intel Core Ultra.
2. **Intel NPU driver** — cari "Intel NPU Driver" di halaman download Intel.

Restart setelah instal.

**Cek lolos:** di Device Manager muncul *Intel(R) Arc(TM) Graphics* (di
Display adapters) dan *Intel(R) AI Boost* (di Neural processors).

---

## Tahap 2 — Ambil kode & setup environment

1. Install **Python 3.12** dari https://www.python.org/downloads/ —
   centang **"Add python.exe to PATH"**.

   ```powershell
   python --version   # harus 3.10 – 3.12
   ```

2. Clone repo:

   ```powershell
   git clone https://github.com/Bengawan-UV-RoboBoat-Team/YOLO-AsusNUC15Pro.git
   cd YOLO-AsusNUC15Pro
   ```

   > Selama perbaikan belum di-merge ke `main`, pindah dulu ke branch-nya:
   > `git checkout fix/model-weights-and-sizes`

3. Jalankan setup (membuat `.venv`, install dependency, cek device):

   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
   ```

   Instalasi torch + openvino cukup besar, bisa 5–15 menit.

**Cek lolos:** baris terakhir setup menampilkan

```
Available OpenVINO devices: ['CPU', 'GPU', 'NPU']
```

Kalau `GPU` atau `NPU` tidak muncul → kembali ke Tahap 1. Benchmark tetap
bisa jalan, tapi device yang hilang akan di-skip dan tidak ada hasilnya.

---

## Tahap 2b — Siapkan semua model di awal (`--prepare`)

`--prepare` mengunduh semua bobot `.pt` + dataset coco128 dan meng-export
semua model ke OpenVINO untuk semua presisi — **tanpa** menjalankan
benchmark. Dengan begitu benchmark tidak butuh internet sama sekali, dan
masalah download/export ketahuan di awal, bukan di tengah sweep.

Pakai filter yang **sama** dengan run yang nanti dijalankan. Untuk
menyiapkan semuanya sekaligus (default run + full sweep):

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark --prepare --sizes n,s,m,l,x --precisions fp32,fp16,int8
```

Butuh ±2–3 GB disk dan bisa makan waktu lama (export INT8 ±2 menit per
model karena kalibrasi). Kalau ada yang gagal, jalankan perintah yang sama
lagi — yang sudah jadi di-skip, hanya yang gagal yang diulang.

Ada dua cara:

**A. Prepare langsung di NUC** (paling simpel, NUC perlu internet saat
tahap ini saja) — jalankan perintah di atas di NUC.

**B. Prepare di laptop lain, lalu copy ke NUC** (NUC tidak perlu internet
sama sekali; export tidak memakan waktu NUC):

1. Jalankan perintah di atas di laptop.
2. Pastikan versi paket di laptop **sama** dengan di NUC — cek di kedua
   mesin:

   ```powershell
   .venv\Scripts\python.exe -c "import ultralytics, openvino, nncf; print(ultralytics.__version__, openvino.__version__, nncf.__version__)"
   ```

   Kalau berbeda, samakan dengan `pip install ultralytics==<versi> openvino==<versi> nncf==<versi>`
   di NUC.
3. Copy folder **`models\`** dan **`data\`** dari laptop ke folder project
   yang sama di NUC (flashdisk / jaringan). Isinya tidak ikut git.

**Cek lolos:** baris terakhir `[prepare] done: N/N export(s) ready ...`
tanpa baris `failed`.

---

## Tahap 3 — Smoke test (±5 menit)

Tes kecil untuk memastikan seluruh alur (download → export → inference →
validasi → CSV) jalan di ketiga device, sebelum menghabiskan waktu berjam-jam
untuk full sweep.

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark --families yolo11 --sizes n --precisions fp32,int8 --devices CPU,GPU,NPU --warmup-iters 5 --timed-iters 30
```

**Cek lolos:**

- Terminal menampilkan `[devices] using CPU / GPU / NPU` (tiga-tiganya).
- Ada 6 kombinasi (`[1/6]` … `[6/6]`), idealnya semua `-> ok`.
- Tabel `=== FPS (mean) ===` dan `=== mAP50-95 ===` tercetak di akhir.
- mAP50-95 untuk yolo11n berada di kisaran **~0.45–0.55** di semua device.
  Kalau satu device punya mAP jauh berbeda (misal < 0.3), ada yang salah
  dengan export/inference di device itu — catat dan laporkan.

Kombinasi NPU yang `FAILED` / `crashed` **boleh terjadi** (lihat Tahap 6),
asalkan CPU dan GPU lolos.

---

## Tahap 4 — Run default (±30–60 menit)

14 model (ukuran n dan s dari semua family) × fp32/int8 × CPU/GPU/NPU:

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark
```

Di awal, terminal mencetak daftar model yang akan dites dan yang
tersaring (`not run (filtered out ...)`) — cek daftarnya masuk akal.

Setelah selesai, **amankan hasilnya** (file summary ditimpa setiap run):

```powershell
Copy-Item results\benchmark_summary.csv results\summary_default_$(Get-Date -Format yyyyMMdd).csv
```

**Cek lolos:** mayoritas baris di CSV berstatus `ok`.

---

## Tahap 5 — Full sweep (opsional, bisa berjam-jam)

Semua ukuran (42 model) × semua presisi × semua device:

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark --sizes n,s,m,l,x --precisions fp32,fp16,int8 --devices CPU,GPU,NPU
```

- Aman ditinggal. Kalau terhenti di tengah (Ctrl+C, mati listrik), hasil
  yang sudah selesai tetap ada di `results/raw/<timestamp>.json`.
- Kalau Tahap 2b sudah dijalankan dengan filter yang sama, tidak ada
  download/export sama sekali — semua langsung diambil dari `models/`.
- Matikan sleep/hibernate otomatis selama run.

Setelah selesai, salin summary seperti di Tahap 4 dengan nama berbeda.

---

## Tahap 6 — Membaca hasil

`results/benchmark_summary.csv` — satu baris per model × presisi × device.

| Kolom | Arti |
|---|---|
| `fps_mean` | Rata-rata FPS (lebih tinggi lebih baik) |
| `latency_ms_mean`, `latency_ms_p95` | Latensi rata-rata dan persentil-95 per gambar |
| `map50_95`, `map50` | Akurasi di coco128 — **indikatif**, hanya untuk perbandingan relatif antar device/presisi |
| `model_size_mb` | Ukuran model OpenVINO di disk |
| `status` | Lihat tabel di bawah |
| `error` | Pesan error kalau gagal |

| Status | Arti | Tindakan |
|---|---|---|
| `ok` | Berhasil | — |
| `export_failed` | Gagal konversi ke OpenVINO (semua device untuk model+presisi itu ikut dilewati) | Cek kolom `error`; biasanya masalah dependency |
| `inference_failed` | Model gagal jalan di device itu (sering: operator tidak didukung NPU) | Wajar untuk NPU; catat sebagai temuan |
| `crashed` | Proses crash native (segfault) di driver/plugin | Wajar sesekali di GPU/NPU; kalau sering, update driver lalu ulangi |

Hal yang menarik dibandingkan:

- **CPU vs GPU vs NPU** untuk model & presisi yang sama.
- **fp32 vs int8** — seberapa besar percepatan dan seberapa turun mAP-nya.
- **Antar family** di ukuran yang sama (misal yolo26n vs yolo11n vs yolov8n).

---

## Tahap 7 — Simpan & laporkan

1. Commit summary CSV yang sudah disalin (folder `results/raw/` tidak
   di-track git — simpan manual kalau perlu datanya):

   ```powershell
   git add results\summary_*.csv
   git commit -m "results: benchmark NUC 15 Pro <tanggal>"
   ```

2. Catat juga di laporan: versi driver Arc & NPU, versi `ultralytics` dan
   `openvino` (ada di kolom `ultralytics_version` CSV / output awal run),
   dan power mode yang dipakai — angka FPS tidak bisa dibandingkan tanpa
   informasi ini.

---

## Troubleshooting

| Gejala | Penyebab & solusi |
|---|---|
| `scripts\setup_env.ps1 cannot be loaded ... running scripts is disabled` | Jalankan dengan `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1` |
| `ModuleNotFoundError: No module named 'yolobench'` | Paket belum terinstal: `.venv\Scripts\python.exe -m pip install -e .` |
| `GPU` / `NPU` tidak ada di daftar device | Driver belum terinstal / perlu restart (Tahap 1) |
| Export INT8 gagal menyebut `nncf` | `.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| FPS sangat rendah & tidak stabil | Belum dicolok listrik / power mode bukan Best performance / ada aplikasi berat |
| Muncul file `kernel.errors.txt` | Dump error compiler kernel driver GPU Intel — tanda driver GPU perlu update. Sudah di-`.gitignore` |
| Banyak baris `crashed` di GPU/NPU | Update driver Arc/NPU ke versi terbaru, lalu ulangi run |
| Benchmark mencoba download padahal sudah `--prepare` | Filter run berbeda dengan filter saat prepare (model/presisi itu belum disiapkan), atau folder `models\`/`data\` tidak ter-copy utuh |
