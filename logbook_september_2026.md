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

---

## Log 13 September 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 13 September 2026 |
| **Rencana Harian** | Mengevaluasi hasil *training* V7 dan merancang arsitektur pamungkas V8 menggunakan strategi *Hierarchical Classification* murni Mod-SE CNN tanpa bantuan *Knowledge Distillation*. |
| **Aktivitas yang Dilakukan** | 1. Menganalisis log V7: Focal Loss terbukti berhasil meningkatkan sensitivitas terhadap kelas radang (Recall MES1 dan MES2 naik drastis menjadi >74%). Namun, tingginya sensitivitas ini menyebabkan banyak *false positive* pada kelas mayoritas (MES0), sehingga akurasi total turun dari 79.40% ke 78.27%.<br>2. Menyusun strategi *Two-Stage Classification* (Hierarkis) untuk V8 dengan tetap mempertahankan arsitektur murni Mod-SE CNN sesuai *constraint* riset.<br>3. Membuat skrip `train_colonomind_v8_hierarchical.py` yang memecah *pipeline* menjadi dua agen: *Detector* (Normal vs Sakit) dan *Severity Grader* (MES1 vs 2 vs 3).<br>4. Menerapkan *Transfer Learning*: Memuat *weights* dari model V6, membongkar *layer* klasifikasinya, dan melakukan *fine-tuning* pada masing-masing tugas spesifik (biner dan terner) dengan *learning rate* sangat kecil. |
| **Hasil/Capaian** | Arsitektur V8 berhasil diselesaikan dan lolos uji sintaks. *Pipeline* prediksi ujung-ke-ujung (*end-to-end*) telah disusun sehingga gambar akan disaring oleh spesialis Normal (Tahap 1) sebelum diteruskan ke spesialis Derajat Keparahan (Tahap 2). |
| **Kendala/Masalah** | Keterbatasan kapasitas ekstraksi fitur dari CNN kecil (Mod-SE CNN) yang menyebabkan saling tumpang tindihnya fitur antara MES0 dan MES1 jika diajarkan 4 kelas sekaligus secara langsung. |
| **Solusi/Tindak Lanjut** | Strategi hierarkis (membagi beban kerja klasifikasi) memungkinkan Mod-SE CNN berkonsentrasi penuh membedakan *subset* visual tertentu tanpa terdistraksi. Siap dieksekusi di *server*. |
| **Dokumentasi (Link/Ref)** | `train_colonomind_v8_hierarchical.py` |

---

## Log 15 September 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 15 September 2026 |
| **Rencana Harian** | Mengeksekusi pelatihan *pipeline* V8 Hierarchical, memecahkan kendala teknis *framework*, dan memastikan seluruh model prediksi (CNN dan Super Agent) tersimpan sempurna. |
| **Aktivitas yang Dilakukan** | 1. Melakukan *troubleshooting* pada *bug* perpindahan bobot Keras (*ValueError: No such layer*) dengan memotong (*slicing*) list array bobot secara langsung menggunakan `get_weights()` untuk mengabaikan auto-inkremen nama lapisan.<br>2. Menyelesaikan kebocoran data (*data leakage*) dan mis-alignment set pengujian pada Agen Tahap 4 dengan menerapkan pemisahan data stratifikasi tunggal (berdasarkan `y_enc_full`) yang sama persis dengan skema V6.<br>3. Menambahkan skrip penyimpanan untuk Standard Scaler, UMAP, serta *booster* dari kedua Super Agent LightGBM agar tidak hilang setelah evaluasi memori. |
| **Hasil/Capaian** | **Terobosan Utama!** Pipeline V8 berhasil berjalan lancar dari ujung-ke-ujung (Tahap 1 hingga Tahap 4). Kombinasi "Detector" (Biner) dan "Grader" (Terner) terbukti ampuh mengatasi bias kelas dan berhasil mengamankan **akurasi gabungan menembus batas >80%** menggunakan arsitektur Mod-SE CNN murni! Seluruh model akhir berhasil diekstraksi ke disk. |
| **Kendala/Masalah** | Penamaan layer otomatis oleh Keras, dan masalah logika pemisahan data uji di awal *development* yang menyebabkan hasil tahap 4 acak (31%). Ketiadaan perintah *save_model* pada agen pohon keputusan di iterasi pertama. |
| **Solusi/Tindak Lanjut** | Merombak logika perpindahan bobot (*weight transfer*), merekayasa ulang alur pemisahan set data, dan memodifikasi *script* sehingga *training* CNN dapat di-*skip* dan langsung menyimpan agen *ensemble* dalam hitungan detik. |
| **Dokumentasi (Link/Ref)** | `train_colonomind_v8_hierarchical.py`, `v8_saving.log` |

---

## Log 16 September 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 16 September 2026 |
| **Rencana Harian** | Mengevaluasi hasil V8 dan merancang arsitektur pamungkas generasi kedua: **ColonoMind V9 (Mod-SE V2)** untuk mengembalikan F1-Score MES1 dan MES2 yang turun akibat efek samping strategi *hierarchical*. |
| **Aktivitas yang Dilakukan** | 1. Menganalisis *trade-off* di V8: Skema hierarkis berhasil mengamankan akurasi total >80% (menekan *false positive* MES0), namun mengorbankan Recall/F1 pada kelas MES1 (68%) dan MES2 (69%).<br>2. Mengevolusikan blok arsitektur kustom *Squeeze-and-Excitation* (SE) menjadi *Convolutional Block Attention Module* (CBAM). Penambahan *Spatial Attention* ini bertujuan agar model mampu melacak **lokasi presisi** lesi mukosa, bukan sekadar warnanya.<br>3. Menanamkan *Residual Connections* (jalan pintas identitas) di setiap *layer* konvolusi utama untuk memungkinkan ekstraksi fitur yang lebih dalam tanpa memicu *vanishing gradient*.<br>4. Menyusun skrip `train_colonomind_v9_modsev2.py` yang melatih arsitektur V2 ini murni dari awal (*from scratch*) dengan penggabungan pelindung Focal Loss (*gamma=2.0*) dan agregator akhir LightGBM Super Agent. |
| **Hasil/Capaian** | Arsitektur *custom* Mod-SE V2 berhasil di-koding dan lolos uji sintaks Keras. Ini memberikan nilai kebaruan (*novelty*) tambahan yang masif untuk manuskrip riset karena membuktikan upaya evolusi mandiri (V1 ke V2) tanpa melanggar batasan larangan penggunaan *pre-trained model* dari luar. |
| **Kendala/Masalah** | Kapasitas Mod-SE V1 yang terlalu kecil (mentok) untuk mempertahankan ketajaman fitur spasial pada inflamasi abu-abu (MES1/MES2) di lingkungan data yang super timpang (*imbalanced*). |
| **Solusi/Tindak Lanjut** | Mod-SE V2 (CBAM + Residual) siap untuk dieksekusi dari awal. Skrip dikonfigurasi untuk menjalankan 3 *seed* ensembel secara penuh. |
| **Dokumentasi (Link/Ref)** | `train_colonomind_v9_modsev2.py` |

---

## Log 17 September 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 17 September 2026 |
| **Rencana Harian** | Memperbaiki kegagalan *training* V9 di server DGX akibat isu kompatibilitas Keras 3 dan memastikan pelestarian (*checkpointing*) model berjalan lancar. |
| **Aktivitas yang Dilakukan** | 1. Menganalisis *crash log* V9 di DGX yang gagal me-*load* model akibat blokade sekuritas TensorFlow 2.16+ (Keras 3) terhadap pembacaan *layer* `Lambda`.<br>2. Mencoba bypass sementara dengan `keras.config.enable_unsafe_deserialization()` dan `safe_mode=False`, namun terkendala *NotImplementedError* karena hilangnya *output_shape inference*.<br>3. Menghapus sepenuhnya lapisan `Lambda` dari *CBAM block*.<br>4. Merancang kelas *Custom Layer* yang proper (`ChannelPooling`) dengan dekorator `@register_keras_serializable` agar arsitektur Mod-SE V2 sepenuhnya *Keras 3-compliant*. |
| **Hasil/Capaian** | Arsitektur Mod-SE V2 kini aman dari gagal simpan/muat (*serialization errors*). Masalah *crash* pada DGX terselesaikan secara permanen sehingga proses *training* V9 dapat berjalan kembali tanpa hambatan. |
| **Kendala/Masalah** | Keras 3 menghapus dukungan *implicit output_shape inference* pada fungsi *Lambda* sehingga model yang mengandung *layer* tersebut tidak dapat dilanjutkan pelatihannya (*resume*). |
| **Solusi/Tindak Lanjut** | Melakukan transisi murni ke *Subclassed Layer* (`ChannelPooling`). Melakukan pembersihan sisa model `.h5` lama yang *corrupt* di DGX dan memulai ulang *training* Mod-SE V2 secara bersih (*from scratch*). |
| **Dokumentasi (Link/Ref)** | `train_colonomind_v9_modsev2.py` |

---

## Log 18 September 2026

| Parameter | Deskripsi |
|---|---|
| **Tanggal & Waktu** | 18 September 2026 |
| **Rencana Harian** | Mengevaluasi performa V9 dan merancang arsitektur pamungkas **ColonoMind V10 (Hierarchical Mod-SE V2)** untuk mengatasi masalah akurasi yang anjlok akibat *Focal Loss*. |
| **Aktivitas yang Dilakukan** | 1. Menganalisis *log* hasil akhir V9: Penggunaan *Focal Loss* tunggal berhasil menaikkan *Recall* MES1 dan MES2 hingga ~75%, namun merusak *Precision* secara masif karena model menjadi rentan salah menebak MES0 (normal) sebagai radang aktif. Akurasi total drop ke 71.77%.<br>2. Merancang strategi V10 dengan mengawinkan kelebihan V8 (*Hierarchical Classification*) dengan kelebihan arsitektur V9 (*Mod-SE V2 / CBAM*).<br>3. Menyusun skrip `train_colonomind_v10_hierarchical_v2.py` yang membagi tugas menjadi dua *stage* (Detector & Grader) menggunakan arsitektur Mod-SE V2 secara murni (tanpa *Focal Loss*), dan menggabungkannya ke dalam *Super Agent LightGBM Hierarkis* di akhir tahap. |
| **Hasil/Capaian** | Arsitektur hibrida ColonoMind V10 berhasil dirampungkan dan siap dieksekusi dari awal (*from scratch*) di *server* DGX. Skema ini secara teoritis akan menyaring bias MES0 di Tahap 1, membiarkan *CBAM Attention* bekerja optimal membedakan derajat radang di Tahap 2. |
| **Kendala/Masalah** | Keterbatasan *Single-Stage Classification* dengan data super timpang: memaksa model mengenali kelas minoritas dengan penalti loss berat (Focal Loss) akan selalu menghasilkan *trade-off* berupa *False Positives* yang tinggi pada kelas mayoritas. |
| **Solusi/Tindak Lanjut** | Meninggalkan sistem klasifikasi 4 kelas secara langsung dan kembali ke sistem terpisah 2-tahap (Deteksi lalu Grading) ala V8. |
| **Dokumentasi (Link/Ref)** | `train_colonomind_v10_hierarchical_v2.py` |
