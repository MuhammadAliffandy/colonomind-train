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
