# Logbook ColonoMind Project — September 2026

## 8-9 September 2026 (Combined)
**Milestone:** Perancangan dan Implementasi ColonoMind v5 (Kembali ke Arsitektur Asli)

- **Analisis V4:** Menemukan bahwa implementasi V4 (yang menggunakan `EfficientNetV2-S`) tidak sesuai dengan batasan penelitian pada paper asli. F1-Maximizer juga gagal menaikkan metrik di V4 karena probabilitas model sudah di titik jenuh (overconfident).
- **Studi Legacy Code:** Melakukan bedah kode pada `Legacy_Notebooks/Utilities/Super Agent.ipynb` untuk memastikan arsitektur asli yang dipakai di awal penelitian. Ditemukan bahwa backbone asli adalah **SE-CNN (Conv2D + SE Block)**, bukan EfficientNet. Selain itu, fitur yang masuk ke agen LightGBM adalah **Deep Fusion Features (512-dim) + Probs + Entropy**, bukan sekadar 4 probabilitas saja.
- **Implementasi V5 (`train_colonomind_v5.py`):** 
  - Membangun ulang backbone Mod-SE CNN (Conv2D 32→64→128→256→512 + SE Block).
  - Menambahkan optimasi kuat untuk menutupi kelemahan CNN tanpa pre-trained ImageNet:
    1. **Ordinal Focal Loss + OHEM** (Hanya belajar dari 70% sampel tersulit dan menghukum kesalahan kelas yang jauh).
    2. **CutMix Augmentation (20%)** untuk mencegah over-memorization.
    3. **3-Phase Training** (Warmup → Mid → Full) total 210 epochs dengan Cosine Annealing.
    4. **Super Agent (643-dim features)** yang dituning dengan **Optuna (30 trials × 5-Fold CV)**.
    5. **Per-class confidence threshold routing** untuk Hybrid Agent.
    6. **F1-Maximizer** via Differential Evolution.
- **Deployment:** Menyiapkan `run_colonomind_v5_pipeline.sh` dan menjalankan eksperimen v5 di server DGX menggunakan `nohup`.

---

## 10 September 2026
**Milestone:** Debugging Pipeline V5 dan Fitur Auto-Resume

- **Analisis Server:** Menemukan bahwa proses training Optuna (Super Agent) terhenti tadi malam karena server DGX mengalami restart tak terduga.
- **Log Cleaning:** Memperbaiki file `train_colonomind_v5.py` dengan menambahkan `warnings.filterwarnings("ignore")` untuk membungkam spam `UserWarning` dari Scikit-Learn saat Optuna melatih 150 model LightGBM.
- **Auto-Resume Feature:** Menambahkan logika pada pipeline agar secara otomatis melewati fase training CNN (Phase 1-3) jika file `best_secnn_v5.h5` sudah terdeteksi ada. Ini mencegah penghapusan hasil training CNN yang sudah berjalan berjam-jam sebelum server mati.
- **Git Push & Rerun:** Melakukan komit dan push perbaikan (`feat: skip CNN training if checkpoint exists to resume safely`), dan menjalankan ulang pipeline di DGX. Proses berhasil dilanjutkan langsung ke ekstraksi fitur dan Optuna Tuning (Step 6 & 7).
