# Logbook ColonoMind Project — September 2026

## Log 8-9 September 2026 (Combined)

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 8 - 9 September 2026 |
| **Rencana Harian** | Mengevaluasi kegagalan model V4 dan merancang ulang arsitektur ColonoMind V5 agar kembali sesuai dengan batasan *paper* (menggunakan *Mod-SE CNN* murni tanpa pre-trained ImageNet). |
| **Aktivitas yang Dilakukan** | 1. Melakukan studi pada `Legacy_Notebooks/Utilities/Super Agent.ipynb` untuk membongkar arsitektur asli.<br>2. Menulis ulang skrip `train_colonomind_v5.py` dengan fondasi SE-CNN (Conv2D + SE Block).<br>3. Menambahkan 9 langkah optimasi (Ordinal Focal Loss, CutMix 20%, Cosine Annealing, dsb) untuk menutupi kelemahan CNN.<br>4. Membangun *Super Agent* LightGBM dengan 643-dimensi fitur (608 Deep Fusion + 28 Handcrafted + 2 UMAP + 4 Probs + 1 Entropy) yang di-tuning via Optuna.<br>5. Menjalankan *pipeline* di *server* DGX. |
| **Hasil/Capaian** | Selesai menyusun skrip *pipeline* V5 yang secara teoritis jauh lebih kuat dari V4 dan mematuhi aturan *research*. Proses *training* berhasil dijalankan menggunakan `nohup`. |
| **Kendala/Masalah** | Model V4 dengan *EfficientNetV2-S* melanggar batasan arsitektur riset. Sementara itu, melatih Mod-SE CNN murni dari nol memakan waktu yang sangat lama dan butuh penanganan *overfitting* yang lebih ketat. |
| **Solusi/Tindak Lanjut** | Memberlakukan 3-Phase Training (30 *warmup* + 60 *mid* + 120 *full*) dengan total 210 *epochs*. Menerapkan *CutMix* dan *OHEM* untuk memaksa model fokus membedakan batas-batas kelas MES yang rumit (terutama MES1). |
| **Dokumentasi (Link/Ref)** | `train_colonomind_v5.py`, `run_colonomind_v5_pipeline.sh` |

---

## Log 10 September 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 10 September 2026 |
| **Rencana Harian** | Memantau hasil *training* V5 di *server* DGX dan memperbaiki *pipeline* jika terjadi *error*. |
| **Aktivitas yang Dilakukan** | 1. Mengecek *log* *training* V5 yang terhenti.<br>2. Menganalisis *log* dan menemukan bahwa proses tersangkut pada tahap *Optuna tuning*.<br>3. Menambahkan kode untuk menyembunyikan (*suppress*) spam `UserWarning` dari modul *Scikit-Learn* di LightGBM.<br>4. Membuat fitur *Auto-Resume* di dalam skrip `train_colonomind_v5.py`.<br>5. Mematikan (kill) *zombie process* lama dan menjalankan ulang *pipeline*. |
| **Hasil/Capaian** | *Pipeline* berhasil berjalan dengan *log* yang bersih. Skrip otomatis melewati fase *training* CNN yang memakan waktu lama karena berhasil memuat *checkpoint* model dari hari sebelumnya, langsung melompat ke tahap *Optuna Tuning* dan *Hybrid Routing*. |
| **Kendala/Masalah** | *Server* DGX mengalami *restart* semalaman (berdasarkan notifikasi sistem) sehingga mematikan proses `nohup` secara paksa di tengah jalan (tepat di Langkah 7/9). |
| **Solusi/Tindak Lanjut** | Menghapus kode `rm -f` pada skrip `.sh` untuk menghindari penghapusan model CNN yang sudah susah payah dilatih. Menambahkan kondisi `if os.path.exists` untuk langsung *load* `best_secnn_v5.h5` jika *server* mati lagi di masa depan. |
| **Dokumentasi (Link/Ref)** | `colonomind_v5_training.log` (Disimpan di `../Result/ColonoMind_v5`) |

---

## Log 11 September 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 11 September 2026 |
| **Rencana Harian** | Mengevaluasi performa ColonoMind V6 dan merancang strategi V7 untuk menembus target akurasi >80%, dengan fokus pada peningkatan skor F1 untuk kelas yang tidak seimbang (MES1 dan MES2). Memverifikasi hasil manuskrip secara komprehensif. |
| **Aktivitas yang Dilakukan** | 1. Melakukan audit komprehensif pada hasil *manuscript* untuk memastikan konsistensi Interval Kepercayaan (CI) dan *confusion matrix*.<br>2. Memperbaiki bug mislabeling kolom metrik (Precision/Recall/F1) dan memisahkan ColonoMind dari tabel perbandingan *baseline*.<br>3. Menganalisis log pelatihan V6 (akurasi 79.40%) yang menunjukkan bahwa bottleneck berada pada skor F1 kelas MES1 dan MES2 akibat distribusi data yang *imbalance*.<br>4. Menyusun skrip `train_colonomind_v7.py` yang menerapkan strategi Focal Loss (CNN), Fine-Tuning dari *checkpoints* V6, serta pembobotan kelas eksponensial di tahap LightGBM. |
| **Hasil/Capaian** | 1. Artefak dokumen hasil dan matriks kebingungan bersih dari anomali matematis dan sudah siap diajukan ke *reviewer*.<br>2. Skrip *training* V7 sudah selesai dibuat, diverifikasi tanpa *syntax error*, dan siap dilatih di *server* DGX. |
| **Kendala/Masalah** | Penurunan signifikansi performa pada "zona abu-abu" inflamasi (MES1 dan MES2) di V6, di mana *loss function* konvensional gagal memaksa model untuk membedakan fitur yang mirip di tengah *class imbalance*. |
| **Solusi/Tindak Lanjut** | Penggunaan `CategoricalFocalCrossentropy` dengan parameter *gamma* dan *alpha* khusus untuk meningkatkan *loss penalty* pada kelas *hard-examples* (MES1 dan MES2). Persiapan *training* V7 di server menggunakan sistem berjalan di belakang layar (*nohup*). |
| **Dokumentasi (Link/Ref)** | `train_colonomind_v7.py`, `Final_Results_Formatted.md`, `audit_inconsistency.py` |
