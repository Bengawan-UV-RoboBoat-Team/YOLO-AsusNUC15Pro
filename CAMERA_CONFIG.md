# Konfigurasi Kamera: Logitech C920 di NUC 15 Pro

Pengaturan kamera yang dibutuhkan supaya kamera benar-benar mengirim
**30 FPS stabil** ke deteksi YOLO, alasan tiap pengaturan, dan cara
membuatnya permanen. Prosedur tes kamera ada di
[TEST_CAMERA.md](TEST_CAMERA.md).

Kamera yang dipakai: **Logitech C920 PRO HD Webcam** (USB ID `046d:08e5`),
muncul sebagai `/dev/video0` (gambar) dan `/dev/video1` (metadata).
Semua perintah di sini untuk **Ubuntu** dan butuh `v4l-utils`
(`sudo apt install v4l-utils`).

---

## Ringkasan

| Pengaturan | Nilai | Diatur lewat | Kenapa |
| --- | --- | --- | --- |
| Format piksel | **MJPG** | `--fourcc MJPG` (skrip) | Di YUYV, 1280×720 hanya 10 FPS; di MJPG 30 FPS |
| Resolusi | **1280×720** | `--width 1280 --height 720` (skrip) | 30 FPS di MJPG; model tetap memakai input 640 |
| `exposure_dynamic_framerate` | **0** (mati) | `v4l2-ctl` | Kalau aktif, kamera **menurunkan FPS** saat cahaya kurang |
| `power_line_frequency` | **1** (50 Hz) | `v4l2-ctl` | Listrik Indonesia 50 Hz; setelan 60 Hz membuat gambar berkedip/bergaris di bawah lampu |
| `auto_exposure` | 3 (Aperture Priority, default) | — | Biarkan otomatis; lihat [opsi exposure manual](#opsi-exposure-manual) untuk kondisi khusus |

Hasil ukur di NUC ini (yolo26s int8 di GPU, 1280×720 MJPG):

| Kondisi | FPS kamera | Frame terlewat |
| --- | ---: | ---: |
| Sebelum (`exposure_dynamic_framerate=1`, 60 Hz), ruangan redup | 17–22 | — |
| Sesudah (`exposure_dynamic_framerate=0`, 50 Hz) | **30,0** | **0%** |
| Kamera mentah tanpa model (`v4l2-ctl --stream-mmap`) | 30,05 | — |

> Default pabrik `exposure_dynamic_framerate` adalah 0, tapi di mesin ini
> nilainya ditemukan 1. Aplikasi lain yang pernah membuka kamera (browser,
> aplikasi video call, dll.) bisa mengubahnya, jadi selalu cek sebelum tes.

---

## 1. Cek pengaturan saat ini

```bash
v4l2-ctl -d /dev/video0 --list-ctrls | grep -E "dynamic_framerate|power_line|auto_exposure|exposure_time|gain"
```

Yang diharapkan:

```text
power_line_frequency ... value=1 (50 Hz)
auto_exposure ... value=3 (Aperture Priority Mode)
exposure_dynamic_framerate ... value=0
```

`gain` yang mendekati 255 (maksimum) menandakan ruangan gelap; kamera
sudah menaikkan penguatan sebanyak mungkin.

---

## 2. Terapkan sekarang (sementara)

```bash
v4l2-ctl -d /dev/video0 -c exposure_dynamic_framerate=0 -c power_line_frequency=1
```

Berlaku langsung, tapi **hilang setiap kali kamera dicabut atau NUC
di-reboot**. Untuk membuatnya otomatis, lihat [bagian 3](#3-buat-permanen-aturan-udev).

---

## 3. Buat permanen (aturan udev)

Aturan udev menjalankan `v4l2-ctl` otomatis setiap kali C920 dicolok atau
NUC menyala.

```bash
sudo tee /etc/udev/rules.d/99-c920.rules > /dev/null <<'EOF'
# Logitech C920 PRO (046d:08e5): 30 FPS tetap di cahaya redup, anti-flicker 50 Hz
ACTION=="add", SUBSYSTEM=="video4linux", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="08e5", ATTR{index}=="0", RUN+="/usr/bin/v4l2-ctl -d $devnode -c exposure_dynamic_framerate=0 -c power_line_frequency=1"
EOF
sudo udevadm control --reload-rules
```

- `ATTR{index}=="0"` membatasi aturan ke device gambar (`/dev/video0`), bukan
  device metadata (`/dev/video1`).
- Kalau memakai **C920 versi lain**, cek ID-nya dengan `lsusb | grep -i c920`
  dan ganti `08e5` (C920 lama umumnya `082d`).

Cek aturannya bekerja: **cabut lalu colok lagi** kamera, kemudian jalankan
perintah di [bagian 1](#1-cek-pengaturan-saat-ini). `exposure_dynamic_framerate`
harus 0 dan `power_line_frequency` harus 1 tanpa menjalankan `v4l2-ctl`
manual.

> Kalau setelah dicolok ulang nilainya belum berubah, kemungkinan aturan
> jalan terlalu cepat sebelum kamera siap. Ganti isi `RUN+=` menjadi
> `/bin/sh -c 'sleep 1; /usr/bin/v4l2-ctl -d $devnode -c exposure_dynamic_framerate=0 -c power_line_frequency=1'`,
> lalu `sudo udevadm control --reload-rules` dan colok ulang.

Untuk menghapus aturan: `sudo rm /etc/udev/rules.d/99-c920.rules && sudo udevadm control --reload-rules`.

---

## 4. Verifikasi FPS kamera

**Kamera saja** (tanpa Python, tanpa model), ±10 detik:

```bash
v4l2-ctl -d /dev/video0 --set-fmt-video=width=1280,height=720,pixelformat=MJPG --stream-mmap --stream-count=300
```

Harus menampilkan **≈30 fps** di setiap baris.

**Dengan model** (tes kamera project ini):

```bash
.venv/bin/python -m yolobench.camera --stream 0:yolo26s:int8:GPU --width 1280 --height 720 --fourcc MJPG --duration 30 --no-show
```

Ringkasan harus menunjukkan `30.0 of 30.0 fps processed, 0.0% frames skipped`.

---

## Opsi exposure manual

Dengan `exposure_dynamic_framerate=0`, waktu exposure otomatis dibatasi
maksimal satu frame (±33 ms), jadi di tempat gelap gambar sedikit lebih
gelap. Untuk robot ini justru lebih baik: frame rate tetap, dan **motion
blur lebih sedikit** saat kapal bergerak.

Kalau butuh exposure yang benar-benar tetap (misalnya supaya kecerahan
tidak berubah-ubah saat kamera menghadap matahari lalu bayangan):

```bash
v4l2-ctl -d /dev/video0 -c auto_exposure=1 -c exposure_time_absolute=100   # manual, 10 ms
```

- Satuan `exposure_time_absolute` adalah 100 µs: `100` = 10 ms, `250` = 25 ms.
  **Jangan lebih dari ±330** (33 ms) supaya tetap 30 FPS.
- Di luar ruangan yang terang, nilai kecil (±20–100) mengurangi blur. Di
  dalam ruangan butuh nilai lebih besar.
- Kembali ke otomatis: `v4l2-ctl -d /dev/video0 -c auto_exposure=3`.

Pengaturan manual ini belum diuji di lapangan; tentukan nilainya saat uji
di lokasi sebenarnya (pencahayaan air sangat berbeda dengan ruangan).

---

## Troubleshooting

| Gejala | Penyebab & solusi |
| --- | --- |
| `source_fps` 17–22 padahal format MJPG 1280×720 | `exposure_dynamic_framerate` masih 1 (kamera memperlambat diri di cahaya redup). Terapkan [bagian 2](#2-terapkan-sekarang-sementara) |
| `source_fps` ±10 di 1280×720 | Kamera memakai YUYV, bukan MJPG. Tambahkan `--fourcc MJPG` |
| Pengaturan hilang setelah reboot / colok ulang | Normal tanpa aturan udev. Pasang [bagian 3](#3-buat-permanen-aturan-udev) |
| Gambar berkedip / ada garis horizontal di bawah lampu | `power_line_frequency` bukan 50 Hz |
| Gambar terlalu gelap setelah `exposure_dynamic_framerate=0` | Wajar di ruangan redup. Tambah cahaya, atau atur [exposure manual](#opsi-exposure-manual) |
| `Device or resource busy` saat `v4l2-ctl --stream-mmap` | Kamera sedang dibuka program lain (tes kamera, browser, node ROS kamera). Tutup dulu |
