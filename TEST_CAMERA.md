# Langkah Pengujian Kamera: Asus NUC 15 Pro

Panduan urut untuk menguji model YOLO dengan kamera sungguhan, dari
satu kamera sampai beberapa kamera sekaligus dengan beban robot. Kerjakan
berurutan; tiap langkah punya **cek lolos**. Jangan lanjut sebelum cek itu
terpenuhi.

Benchmark di [TEST_STEP.md](TEST_STEP.md) hanya mengukur satu model pada
satu gambar diam, di mesin yang menganggur. Tes kamera ini menjawab
pertanyaan yang lebih dekat dengan kondisi lomba: **apakah model yang
dipilih tetap sanggup 30 FPS di kamera asli, bersama kamera lain, tracking,
dan software robot lainnya?**

Semua perintah dijalankan dari root folder project di **bash** (Ubuntu
24.04). Di Windows, ganti `.venv/bin/python` dengan
`.venv\Scripts\python.exe`. Langkah yang khusus Ubuntu ditandai.

---

## Tahap 0: Prasyarat

1. **TEST_STEP.md Tahap 0–2 sudah lolos**: driver terpasang, `.venv` sudah
   dibuat, dan OpenVINO mendeteksi `['CPU', 'GPU', 'NPU']`.
2. **Model yang mau dites sudah di-export** (Tahap 2b `--prepare`). Cek:

   ```bash
   .venv/bin/python -m yolobench.camera --list
   ```

   Model yang belum ada di daftar tetap bisa dipakai, tapi akan di-export
   dulu saat tes dimulai (perlu internet, int8 ±2 menit per model).
3. **NUC tersambung listrik** dan power mode **Performance**
   (`powerprofilesctl get` menampilkan `performance`).
4. **Kalau mau memakai `--track`** (Tahap 5), pasang paket `lap` sekali:

   ```bash
   .venv/bin/pip install lap
   ```

   Kalau dilewati, Ultralytics akan mencoba memasangnya otomatis saat
   `--track` pertama kali dipakai (perlu internet).

**Cek lolos:** `--list` menampilkan model yang mau dites, misalnya
`yolo26s`, `yolo26m`, dan `yolo26l` dengan `fp32, int8`.

---

## Tahap 1: Pastikan kamera terdeteksi

1. Colok kamera **langsung ke port USB 3 NUC** (biru / bertanda SS), bukan
   lewat hub pasif. Dua kamera di satu hub sering kekurangan bandwidth.
2. **Ubuntu:** cek device kamera.

   ```bash
   ls /dev/video*
   sudo apt install v4l-utils      # sekali saja, untuk v4l2-ctl
   v4l2-ctl --list-devices
   ```

   Satu kamera USB biasanya muncul sebagai **dua** device, misalnya
   `/dev/video0` dan `/dev/video1`. Hanya yang pertama (angka genap:
   0, 2, 4, ...) yang mengeluarkan gambar; yang kedua berisi metadata. Jadi
   dua kamera biasanya menjadi indeks **0** dan **2**.
3. **Ubuntu:** cek mode yang didukung kamera, terutama apakah 30 FPS
   tersedia di resolusi yang mau dipakai.

   ```bash
   v4l2-ctl -d /dev/video0 --list-formats-ext
   ```

   Perhatikan baris `'MJPG'` dan `'YUYV'`. Banyak kamera hanya memberi
   30 FPS di 720p ke atas dalam format **MJPG**; di YUYV sering hanya 5–10
   FPS.
4. **Windows:** kamera terdaftar di Device Manager → *Cameras*. Indeksnya
   0, 1, 2, ... sesuai urutan colok.

**Cek lolos:** setiap kamera punya indeks yang diketahui, dan mode 30 FPS
di resolusi yang dipilih tersedia (Ubuntu: terlihat di `--list-formats-ext`).

---

## Tahap 2: Satu kamera, model ringan (±1 menit)

Tujuannya memastikan kamera benar-benar memberi 30 FPS. Pakai model paling
ringan supaya yang diuji hanya kameranya.

```bash
.venv/bin/python -m yolobench.camera --stream 0:yolo26n:fp32:GPU --width 1280 --height 720 --fourcc MJPG --duration 60
```

Jendela preview dengan kotak deteksi akan muncul. Arahkan kamera ke orang
atau benda sehari-hari (kelas COCO) supaya ada yang terdeteksi.

**Cek lolos:**

- Di ringkasan akhir, `... of 30.0 fps processed`: angka kedua (FPS
  kamera) **≈ 30**. Kalau jauh di bawah 30 (misalnya 10 atau 15), masalahnya
  di kamera, bukan di model: coba mode lain dari Tahap 1 (`--fourcc MJPG`,
  resolusi lebih kecil), lalu ulangi.
- Status stream `-> OK`.
- File `results/camera/<timestamp>.csv` tersimpan.

---

## Tahap 3: Satu kamera, model kandidat (±1 menit per model)

Ulangi dengan model yang benar-benar mau dipakai. Mulai dari rekomendasi
per batas FPS di README (bagian *Best model per FPS budget*):

```bash
.venv/bin/python -m yolobench.camera --stream 0:yolo26s:int8:GPU --width 1280 --height 720 --fourcc MJPG --duration 60
.venv/bin/python -m yolobench.camera --stream 0:yolo26m:int8:GPU --width 1280 --height 720 --fourcc MJPG --duration 60
.venv/bin/python -m yolobench.camera --stream 0:yolo26l:int8:GPU --width 1280 --height 720 --fourcc MJPG --duration 60
```

Tanpa `--stream`, model, presisi, dan device bisa dipilih dari menu:

```bash
.venv/bin/python -m yolobench.camera --width 1280 --height 720 --fourcc MJPG --duration 60
```

**Cek lolos:** model yang mau dipakai berstatus `-> OK`, dan `inference
p95` masih jauh di bawah `budget 33.3 ms`. Sisakan ruang minimal ±40% (p95
≤ ±20 ms), karena tahap berikutnya menambah beban.

---

## Tahap 4: Beberapa kamera sekaligus (±2 menit)

Semua stream berjalan bersamaan, jadi di sini terlihat apakah GPU/NPU dan
CPU masih sanggup. Konfigurasi yang disarankan untuk 2 kamera: satu di GPU,
satu di NPU, supaya beban terbagi.

```bash
.venv/bin/python -m yolobench.camera \
  --stream 0:yolo26s:int8:GPU \
  --stream 2:yolo26s:fp32:NPU \
  --width 1280 --height 720 --fourcc MJPG --duration 120
```

- Di NPU pakai **fp32**, jangan int8. Untuk yolo26s, int8 di NPU
  menurunkan mAP dari 0.487 ke 0.426 (lihat README).
- Mau membandingkan dengan dua kamera sama-sama di GPU? Ganti stream kedua
  menjadi `2:yolo26s:int8:GPU`.
- Kamera yang sama boleh dipakai dua stream (misalnya
  `--stream 0:yolo26s:int8:GPU --stream 0:yolo26s:fp32:NPU`) untuk
  mensimulasikan dua kamera kalau baru punya satu.

**Cek lolos:** semua stream `-> OK`, masing-masing ≈ 30 FPS terproses dan
frame terlewat ≤ 5%.

---

## Tahap 5: Tambah tracking (±2 menit)

Ulangi konfigurasi Tahap 4 dengan `--track` (ByteTrack):

```bash
.venv/bin/python -m yolobench.camera \
  --stream 0:yolo26s:int8:GPU \
  --stream 2:yolo26s:fp32:NPU \
  --width 1280 --height 720 --fourcc MJPG --duration 120 --track
```

Di preview, setiap objek sekarang punya nomor ID yang seharusnya tetap sama
selama objek itu terlihat.

**Cek lolos:** semua stream masih `-> OK`. Tracking berjalan di CPU, jadi
bandingkan `inference p95` dengan Tahap 4 untuk melihat tambahannya.

---

## Tahap 6: Tes ketahanan dengan beban robot (30 menit)

Tahap paling penting sebelum lomba. Jalankan konfigurasi final **bersamaan**
dengan software robot lainnya (ROS, path planning, node sensor, logging),
dalam kondisi yang semirip mungkin dengan lomba.

1. Jalankan semua software robot seperti biasa.
2. Di terminal lain, jalankan tes kamera selama 30 menit. Lewat SSH atau
   tanpa monitor, tambahkan `--no-show`:

   ```bash
   .venv/bin/python -m yolobench.camera \
     --stream 0:yolo26s:int8:GPU \
     --stream 2:yolo26s:fp32:NPU \
     --width 1280 --height 720 --fourcc MJPG --duration 1800 --track --no-show
   ```

3. Matikan sleep/suspend otomatis selama tes (Ubuntu: Settings → Power →
   Automatic Suspend → Off).
4. Setiap 5 detik terminal mencetak FPS dan p95 per stream, beban CPU, dan
   suhu CPU. Perhatikan apakah angkanya **menurun seiring waktu**, karena itu
   tanda thermal throttling.

**Cek lolos:**

- Semua stream `-> OK` di ringkasan akhir.
- `frame-to-result p99` masih di bawah batas yang bisa diterima robot
  (misalnya ≤ 50 ms).
- FPS di menit-menit terakhir tidak turun dibanding menit pertama.
- `max CPU temp` tidak terus-menerus mendekati ±100 °C, di sekitar suhu itu
  CPU mulai menurunkan kecepatannya (throttling).

---

## Tahap 7: Membaca hasil & menentukan pilihan

Ringkasan di terminal dan file `results/camera/<timestamp>.csv` berisi satu
baris per stream:

| Kolom | Arti |
| --- | --- |
| `source_fps` | FPS yang benar-benar dikirim kamera |
| `processed_fps` | FPS yang berhasil diproses model |
| `skipped_pct` | Persentase frame kamera yang terlewat karena model belum selesai |
| `infer_ms_mean`, `infer_ms_p95`, `infer_ms_p99` | Waktu satu kali deteksi (+ tracking kalau `--track`) |
| `e2e_ms_p95`, `e2e_ms_p99` | Waktu dari frame diterima sampai hasil keluar: umur informasi yang dipakai robot |
| `frame_budget_ms` | Waktu per frame kamera (33.3 ms di 30 FPS) |
| `verdict` | `OK`, atau `TOO SLOW` kalau frame terlewat > 5% atau `infer_ms_p95` > `frame_budget_ms` |
| `cpu_load_mean`, `cpu_temp_max` | Beban CPU rata-rata dan suhu CPU tertinggi selama tes |

Cara memilih:

1. Buang konfigurasi yang `TOO SLOW` di Tahap 6.
2. Dari yang `OK`, pilih model paling besar (paling akurat) yang
   `infer_ms_p95`-nya masih menyisakan ruang ±30–40% dari
   `frame_budget_ms`. Ruang itu disiapkan untuk lonjakan beban saat lomba.
3. Kalau tidak ada yang lolos, pilihannya: model lebih kecil, pindahkan satu
   kamera ke NPU, atau jalankan deteksi setiap 2 frame dan biarkan tracker
   mengisi frame di antaranya.

Catat konfigurasi final beserta file CSV Tahap 6-nya, lalu commit:

```bash
git add results/camera/<timestamp>.csv
git commit -m "results: camera test NUC 15 Pro <konfigurasi>"
```

---

## Tanpa kamera: tes dengan file video

Semua tahap di atas bisa dicoba dengan file video sebagai pengganti kamera.
Video diputar sesuai frame rate aslinya, jadi perilakunya seperti kamera
(rekaman lintasan atau kolam paling ideal):

```bash
.venv/bin/python -m yolobench.camera --stream rekaman.mp4:yolo26s:int8:GPU
```

Tes berhenti otomatis saat video habis. Untuk tes ketahanan, pakai video
yang cukup panjang.

---

## Troubleshooting

| Gejala | Penyebab & solusi |
| --- | --- |
| `can't open source '0'` | Indeks salah atau kamera belum terdeteksi. Ubuntu: cek `v4l2-ctl --list-devices`, coba indeks genap berikutnya (2, 4, ...). Kamera sedang dipakai aplikasi lain (browser, Zoom, node ROS kamera)? Tutup dulu |
| `source_fps` jauh di bawah 30 | Mode kamera: pakai `--fourcc MJPG`, resolusi lebih kecil, atau cek mode yang didukung (`--list-formats-ext`). Pencahayaan gelap juga bisa membuat kamera menurunkan FPS (auto exposure) |
| Dua kamera: satu gagal dibuka atau FPS-nya drop | Bandwidth USB kurang. Colok kedua kamera ke port USB 3 yang berbeda langsung di NUC, pakai MJPG, atau turunkan resolusi |
| `device(s) not available on this machine: NPU` | Driver NPU / grup `render` belum beres, lihat TEST_STEP.md Tahap 1 |
| `unknown model '...'` | Nama model salah ketik. Lihat daftarnya dengan `--list` |
| `--track` error menyebut `lap` | Paket belum terpasang: `.venv/bin/pip install lap` |
| Jendela preview tidak muncul / error `display` | Dijalankan lewat SSH tanpa layar. Tambahkan `--no-show` |
| `TOO SLOW` padahal di benchmark model itu cepat | Wajar: tes ini menghitung beban stream lain, CPU, dan tracking. Lihat Tahap 7 langkah 3 |
| FPS turun setelah beberapa menit | Thermal throttling. Cek suhu di output, pastikan ventilasi NUC tidak tertutup |
| Suhu CPU tidak tampil | Normal di Windows (psutil tidak bisa membaca suhu di Windows) |
