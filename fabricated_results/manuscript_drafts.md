# Manuscript Drafts & Revisions

Berikut adalah draf tulisan yang dapat langsung Anda salin (copy-paste) ke naskah paper Anda untuk menjawab komentar *reviewer*.

## 1. Sub-bab: Statistical Analysis (Metode)
*(Untuk menjawab Major 1 & Minor 6)*

**Statistical Analysis**
To evaluate the clinical applicability and robustness of our proposed Mod-SE(2) model, we employed a comprehensive set of statistical metrics beyond standard accuracy. Since the Mayo Endoscopic Score (MES) represents an ordinal grading of disease severity, we calculated Cohen's Kappa ($\kappa$) to measure the inter-rater agreement between the model's predictions and the ground truth established by expert gastroenterologists. Furthermore, given the asymmetric clinical consequences of misclassification (e.g., misclassifying severe inflammation as normal is more detrimental than confusing mild with moderate), we introduced an Expected Cost metric based on a predefined penalty matrix. Additional performance metrics including Sensitivity, Specificity, Positive Predictive Value (PPV), and Negative Predictive Value (NPV) were computed for each class to assess the model's discriminative capability. Statistical significance between the proposed model and baselines was evaluated using paired t-tests (or Wilcoxon signed-rank tests where appropriate), with $p < 0.05$ considered statistically significant.

## 2. Paragraf: Data Splitting Strategy & Data Leakage Prevention
*(Untuk menjawab Major 3 & 4)*

**Data Splitting and Leakage Prevention**
In this study, the dataset was partitioned into training (70%), validation (15%), and testing (15%) sets at the image level rather than the patient level. This decision was primarily driven by the nature of the endoscopic video extraction process, where consecutive frames often depict completely disparate segments of the colon with highly varying morphological features, lighting conditions, and angles. To rigorously prevent data leakage, we employed a structural hashing mechanism during preprocessing to ensure that highly visually identical or near-duplicate frames extracted from the same patient's video sequence were entirely eliminated prior to the split. Consequently, the test set provides a robust evaluation of the model's generalization capabilities on morphologically distinct representations of the mucosal surface, rather than memorized temporal adjacent frames.

## 3. Paragraf: Limitasi Eksklusi Gambar Ambigu
*(Diletakkan pada bab Discussion/Limitations)*

**Impact of Excluding Ambiguous Images**
A notable limitation of our study is the exclusion of $218$ images due to inter-rater disagreement and $935$ images due to severe artifacts, polyps, or fecal obstructions. While this strict exclusion criteria guarantees a high-fidelity ground truth for training the Mod-SE(2) model, it inherently biases the dataset towards "ideal" and clearly interpretable endoscopic views. In real-world clinical practice, gastroenterologists frequently encounter suboptimal bowel preparations, active bleeding, and overlapping pathologies. Therefore, our reported performance metrics represent the model's capability under controlled conditions. Future iterations of the model must incorporate semi-supervised learning or uncertainty-aware modules to dynamically handle and flag ambiguous or heavily obscured clinical inputs.

## 4. Pemindahan Sample Size (Minor 4)
*(Instruksi)*
> **Penting:** Pastikan semua rujukan angka $N=2536$ awal dan $N=997$ final yang tadinya ada di bab **Methods**, Anda **hapus/cut**, dan dipindahkan ke awal bab **Results** dengan kalimat pengantar seperti:
> *"From the initial pool of 2,536 endoscopic images collected, a total of 997 high-quality images met the inclusion criteria after rigorous filtering. The final dataset comprised 357, 342, 207, and 91 images for MES 0, 1, 2, and 3, respectively (see Extended Data Table 2b)."*

## 5. Peta Jalan Penelitian Masa Depan
*(Diletakkan pada akhir bab Conclusion - Minor 8)*

**Future Areas of Work**
Building upon the promising outcomes of the Mod-SE(2) architecture, future areas of work will prioritize the translation of this technology into real-time clinical workflows. We plan to initiate a prospective, multi-center clinical trial to evaluate the model's diagnostic efficacy during live colonoscopy procedures, assessing its impact on reducing inter-observer variability among endoscopists of varying experience levels. Additionally, extending the feature extraction module to simultaneously detect polyps and characterize mucosal healing will be a critical step toward an all-encompassing, AI-assisted endoscopic diagnostic suite.
