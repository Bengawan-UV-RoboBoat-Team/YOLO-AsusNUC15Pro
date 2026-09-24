# Langkah Pengujian: Asus NUC 15 Pro

Panduan urut untuk menjalankan benchmark YOLO di NUC 15 Pro, dari mesin
kosong sampai hasil tersimpan. Kerjakan berurutan; tiap langkah punya
**cek lolos**. Jangan lanjut sebelum cek itu terpenuhi.

Panduan ini berlaku untuk **Windows 11** dan **Ubuntu 24.04**. Semua
perintah dijalankan dari root folder project: di **PowerShell** untuk
Windows, di **bash** untuk Ubuntu. Contoh perintah ditulis versi Windows;
di Ubuntu, ganti `.venv\Scripts\python.exe` dengan `.venv/bin/python`.
Langkah yang berbeda di Ubuntu ditulis terpisah.

---

## Tahap 0: Persiapan mesin

1. **Colok NUC ke listrik** dan set power mode ke performa maksimal.
   Mode hemat daya men-throttle CPU/GPU dan membuat FPS jauh lebih rendah.
   - Windows: Settings → System → Power → Power mode → **Best performance**.
   - Ubuntu: Settings → Power → Power Mode → **Performance**, atau
     `powerprofilesctl set performance`.

   NUC tidak punya baterai, jadi cek adaptor listrik secara fisik. Di
   Ubuntu, nilai `/sys/class/power_supply/*/online` bisa berasal dari port
   USB-C dan tidak menunjukkan status adaptor.
2. **Tutup aplikasi berat lain** (browser dengan banyak tab, game, sync
   OneDrive besar) supaya tidak mengganggu pengukuran.
3. **Update OS** dan restart sekali sebelum mulai.
   - Windows: Windows Update.
   - Ubuntu: `sudo apt update && sudo apt full-upgrade`.

**Cek lolos:** NUC tersambung listrik, power mode = Best performance
(Windows) / Performance (Ubuntu: `powerprofilesctl get` menampilkan
`performance`).

---

## Tahap 1: Driver

Driver iGPU dan NPU **terpisah**, keduanya harus diinstal.

### Windows

1. **Intel Arc Graphics driver** (untuk iGPU): dari halaman download Intel,
   pilih driver terbaru untuk Intel Core Ultra.
2. **Intel NPU driver**: cari "Intel NPU Driver" di halaman download Intel.

Restart setelah instal.

**Cek lolos:** di Device Manager muncul *Intel(R) Arc(TM) Graphics* (di
Display adapters) dan *Intel(R) AI Boost* (di Neural processors).

### Ubuntu 24.04

1. **Kernel HWE**: kernel 6.8 bawaan 24.04 bisa jadi terlalu lama untuk
   iGPU/NPU Core Ultra seri 200. Cek dulu dengan `uname -r`; kalau sudah
   6.11 atau lebih baru, lewati langkah ini.

   ```bash
   sudo apt install linux-generic-hwe-24.04
   ```

2. **Driver iGPU** (compute runtime OpenCL/Level Zero): `sudo apt install
   intel-opencl-icd`, atau paket `.deb` terbaru dari
   <https://github.com/intel/compute-runtime/releases> (lebih disarankan
   untuk chip baru). Kalau memakai `.deb`, paket compiler IGC
   (`intel-igc-core-2`, `intel-igc-opencl-2`) dari
   <https://github.com/intel/intel-graphics-compiler/releases> juga wajib
   diunduh. Link `wget` lengkapnya ada di catatan rilis compute-runtime.
   Taruh semua `.deb` di satu folder kosong, lalu:

   ```bash
   sudo dpkg -i *.deb
   ```

3. **Driver NPU**: dari <https://github.com/intel/linux-npu-driver/releases>.
   Ringkasan urutan dari catatan rilisnya (ambil nama file & link persisnya
   dari halaman rilis terbaru):

   ```bash
   # hapus paket NPU versi lama (aman kalau belum ada)
   sudo dpkg --purge --force-remove-reinstreq intel-driver-compiler-npu intel-fw-npu intel-level-zero-npu intel-level-zero-npu-dbgsym
   # unduh & ekstrak arsip untuk Ubuntu 24.04 (file ...-ubuntu2404.tar.gz)
   tar -xf linux-npu-driver-<versi>-ubuntu2404.tar.gz
   sudo apt update && sudo apt install libtbb12
   sudo dpkg -i *.deb
   # pasang Level Zero loader (libze1) dari link di catatan rilis
   sudo dpkg -i libze1_*.deb
   ```

4. **Grup `render`**: tanpa ini GPU/NPU sering tidak terdeteksi.

   ```bash
   sudo usermod -aG render $USER
   ```

Reboot setelah semua terpasang (sekaligus mengaktifkan kernel HWE dan grup
`render`).

**Cek lolos:**

```bash
uname -r          # 6.11 atau lebih baru
ls /dev/dri/      # ada renderD128 (iGPU)
ls /dev/accel/    # ada accel0 (NPU)
groups            # ada render
dpkg -l | grep -E "intel-opencl-icd|libze-intel-gpu1|intel-level-zero-npu|libze1"
                  # keempat paket muncul (driver userspace iGPU & NPU)
```

`renderD128` dan `accel0` bisa sudah ada walaupun driver userspace belum
terpasang, karena keduanya dibuat oleh kernel. Karena itu cek `dpkg` di
atas tetap wajib; tanpa paket-paket itu OpenVINO tidak akan melihat `GPU`
/ `NPU`.

---

## Tahap 2: Ambil kode & setup environment

1. Siapkan **Python 3.12**.
   - Windows: install dari <https://www.python.org/downloads/> dan centang
     **"Add python.exe to PATH"**.
   - Ubuntu: Python 3.12 sudah bawaan, tinggal pasang modul venv dan git:
     `sudo apt install python3-venv git`.

   ```powershell
   python --version    # Windows, harus 3.10 – 3.12
   python3 --version   # Ubuntu, harus 3.10 – 3.12
   ```

2. Clone repo:

   ```powershell
   git clone https://github.com/Bengawan-UV-RoboBoat-Team/YOLO-AsusNUC15Pro.git
   cd YOLO-AsusNUC15Pro
   ```

3. Jalankan setup (membuat `.venv`, install dependency, cek device):

   ```powershell
   # Windows
   powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1
   ```

   ```bash
   # Ubuntu
   bash scripts/setup_env.sh
   ```

   Instalasi torch + openvino cukup besar, bisa 5–15 menit.

**Cek lolos:** baris terakhir setup menampilkan

```text
Available OpenVINO devices: ['CPU', 'GPU', 'NPU']
```

Kalau `GPU` atau `NPU` tidak muncul → kembali ke Tahap 1. Di Ubuntu,
`setup_env.sh` juga memberi peringatan kalau user belum masuk grup `render`. Benchmark tetap
bisa jalan, tapi device yang hilang akan di-skip dan tidak ada hasilnya.

---

## Tahap 2b: Siapkan semua model di awal (`--prepare`)

`--prepare` mengunduh semua bobot `.pt`, dataset coco128 (untuk kalibrasi
INT8) dan subset COCO val2017 (untuk mAP, ±1 GB), lalu meng-export
semua model ke OpenVINO untuk semua presisi, **tanpa** menjalankan
benchmark. Dengan begitu benchmark tidak butuh internet sama sekali, dan
masalah download/export ketahuan di awal, bukan di tengah sweep.

Pakai filter yang **sama** dengan run yang nanti dijalankan. Untuk
menyiapkan semuanya sekaligus (default run + full sweep):

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark --prepare --sizes n,s,m,l,x --precisions fp32,fp16,int8
```

Butuh ±3–4 GB disk dan bisa makan waktu lama (export INT8 ±2 menit per
model karena kalibrasi). Kalau ada yang gagal, jalankan perintah yang sama
lagi: yang sudah jadi di-skip, hanya yang gagal yang diulang.

Ada dua cara:

**A. Prepare langsung di NUC** (paling simpel, NUC perlu internet saat
tahap ini saja): jalankan perintah di atas di NUC.

**B. Prepare di laptop lain, lalu copy ke NUC** (NUC tidak perlu internet
sama sekali; export tidak memakan waktu NUC):

1. Jalankan perintah di atas di laptop.
2. Pastikan versi paket di laptop **sama** dengan di NUC. Cek di kedua
   mesin:

   ```powershell
   .venv\Scripts\python.exe -c "import ultralytics, openvino, nncf; print(ultralytics.__version__, openvino.__version__, nncf.__version__)"
   ```

   Kalau berbeda, samakan dengan `pip install ultralytics==<versi> openvino==<versi> nncf==<versi>`
   di NUC.
3. Copy folder **`models\`** dan **`data\`** dari laptop ke folder project
   yang sama di NUC (flashdisk / jaringan). Isinya tidak ikut git. Laptop
   dan NUC boleh beda OS (misal prepare di laptop Windows, benchmark di NUC
   Ubuntu): model OpenVINO tidak tergantung OS, dan path dataset di
   `data/ultralytics_config/` serta `data/coco-val500.yaml` ditulis ulang
   otomatis setiap run.

**Cek lolos:** baris terakhir `[prepare] done: N/N export(s) ready ...`
tanpa baris `failed`.

---

## Tahap 3: Smoke test (±5 menit)

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
- Header menampilkan `validation dataset: coco-val500`.
- mAP50-95 untuk yolo11n berada di kisaran **~0.35–0.43** di semua device
  (angka resmi di val2017 penuh: 0.395). Versi lama yang memakai coco128
  menghasilkan ±0.50; angka itu terlalu tinggi karena coco128 adalah data
  latihan. Kalau satu device punya mAP jauh berbeda (misal < 0.25), ada yang
  salah dengan export/inference di device itu. Catat dan laporkan.

Kombinasi NPU yang `FAILED` / `crashed` **boleh terjadi** (lihat Tahap 6),
asalkan CPU dan GPU lolos.

---

## Tahap 4: Run default (±30–60 menit)

14 model (ukuran n dan s dari semua family) × fp32/int8 × CPU/GPU/NPU:

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark
```

Di awal, terminal mencetak daftar model yang akan dites dan yang
tersaring (`not run (filtered out ...)`). Cek apakah daftarnya masuk akal.

Setelah selesai, **amankan hasilnya** (file summary ditimpa setiap run):

```powershell
# Windows
Copy-Item results\benchmark_summary.csv results\summary_default_$(Get-Date -Format yyyyMMdd).csv
```

```bash
# Ubuntu
cp results/benchmark_summary.csv results/summary_default_$(date +%Y%m%d).csv
```

**Cek lolos:** mayoritas baris di CSV berstatus `ok`.

---

## Tahap 5: Full sweep (opsional, bisa berjam-jam)

Semua ukuran (42 model) × semua presisi × semua device:

```powershell
.venv\Scripts\python.exe -m yolobench.benchmark --sizes n,s,m,l,x --precisions fp32,fp16,int8 --devices CPU,GPU,NPU
```

- Aman ditinggal. Kalau terhenti di tengah (Ctrl+C, mati listrik), hasil
  yang sudah selesai tetap ada di `results/raw/<timestamp>.json`.
- Kalau Tahap 2b sudah dijalankan dengan filter yang sama, tidak ada
  download/export sama sekali karena semua langsung diambil dari `models/`.
- Matikan sleep/hibernate otomatis selama run (Ubuntu: Settings → Power →
  Automatic Suspend → Off).

Setelah selesai, salin summary seperti di Tahap 4 dengan nama berbeda.

---

## Tahap 6: Membaca hasil

`results/benchmark_summary.csv`: satu baris per model × presisi × device.

| Kolom | Arti |
| --- | --- |
| `fps_mean` | Rata-rata FPS (lebih tinggi lebih baik) |
| `latency_ms_mean`, `latency_ms_p95` | Latensi rata-rata dan persentil-95 per gambar |
| `map50_95`, `map50` | Akurasi di subset 500 gambar COCO val2017 (data yang tidak dipakai untuk melatih model) |
| `val_dataset` | Dataset tempat mAP diukur (`coco-val500`; hasil lama: `coco128`) |
| `model_size_mb` | Ukuran model OpenVINO di disk |
| `status` | Lihat tabel di bawah |
| `error` | Pesan error kalau gagal |

| Status | Arti | Tindakan |
| --- | --- | --- |
| `ok` | Berhasil | - |
| `export_failed` | Gagal konversi ke OpenVINO (semua device untuk model+presisi itu ikut dilewati) | Cek kolom `error`; biasanya masalah dependency |
| `inference_failed` | Model gagal jalan di device itu (sering: operator tidak didukung NPU) | Wajar untuk NPU; catat sebagai temuan |
| `crashed` | Proses crash native (segfault) di driver/plugin | Wajar sesekali di GPU/NPU; kalau sering, update driver lalu ulangi |

Hal yang menarik dibandingkan:

- **CPU vs GPU vs NPU** untuk model & presisi yang sama.
- **fp32 vs int8**: seberapa besar percepatan dan seberapa turun mAP-nya.
- **Antar family** di ukuran yang sama (misal yolo26n vs yolo11n vs yolov8n).

---

## Tahap 7: Simpan & laporkan

1. Commit summary CSV yang sudah disalin (folder `results/raw/` tidak
   di-track git, jadi simpan manual kalau perlu datanya):

   ```powershell
   git add results/summary_*.csv
   git commit -m "results: benchmark NUC 15 Pro <tanggal>"
   ```

2. Catat juga di laporan: OS (Windows 11 / Ubuntu 24.04 + versi kernel
   dari `uname -r`), versi driver Arc & NPU, versi `ultralytics` dan
   `openvino` (ada di kolom `ultralytics_version` CSV / output awal run),
   dan power mode yang dipakai, karena angka FPS tidak bisa dibandingkan tanpa
   informasi ini.

---

## Troubleshooting

| Gejala | Penyebab & solusi |
| --- | --- |
| `scripts\setup_env.ps1 cannot be loaded ... running scripts is disabled` | Jalankan dengan `powershell -ExecutionPolicy Bypass -File scripts\setup_env.ps1` |
| `ModuleNotFoundError: No module named 'yolobench'` | Paket belum terinstal: `.venv\Scripts\python.exe -m pip install -e .` |
| `GPU` / `NPU` tidak ada di daftar device | Driver belum terinstal / perlu restart (Tahap 1) |
| (Ubuntu) `GPU` / `NPU` tetap tidak muncul padahal driver sudah terpasang | User belum di grup `render` atau belum logout/login ulang (cek `groups`); atau kernel masih 6.8 (cek `uname -r`), pasang `linux-generic-hwe-24.04` lalu reboot |
| (Ubuntu) `/dev/accel/accel0` / `/dev/dri/renderD128` ada tapi `NPU` / `GPU` tidak muncul di OpenVINO | Driver userspace belum terpasang: paket NPU + `libze1`, atau compute runtime iGPU (Tahap 1, cek dengan `dpkg -l`) |
| (Ubuntu) `ensurepip is not available` / gagal membuat `.venv` | Modul venv belum ada: `sudo apt install python3-venv`, hapus `.venv`, jalankan setup lagi |
| (Ubuntu) `$'\r': command not found` saat menjalankan `setup_env.sh` | File ter-copy dengan line ending Windows. Ambil lewat `git clone`/`git pull`, atau perbaiki dengan `sed -i 's/\r$//' scripts/setup_env.sh` |
| Export INT8 gagal menyebut `nncf` | `.venv\Scripts\python.exe -m pip install -r requirements.txt` |
| FPS sangat rendah & tidak stabil | Belum dicolok listrik / power mode bukan Best performance / ada aplikasi berat |
| Muncul file `kernel.errors.txt` | Dump error compiler kernel driver GPU Intel dan menandakan driver GPU perlu update. Sudah di-`.gitignore` |
| Banyak baris `crashed` di GPU/NPU | Update driver Arc/NPU ke versi terbaru, lalu ulangi run |
| Benchmark mencoba download padahal sudah `--prepare` | Filter run berbeda dengan filter saat prepare (model/presisi itu belum disiapkan), atau folder `models\`/`data\` tidak ter-copy utuh |
