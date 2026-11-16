# UTS Deep Learning — Tugas Kelompok

**"Hands-on Deep Learning: Classification, Regression, and Clustering"**

> Salinan teks soal dari dosen (bahasa asli). Jangan diubah isinya; catatan kita taruh di `docs/rencana.md`.

## 1. Main Objective

To assess students' ability to build complete and honest Deep Learning pipelines for classification, regression, and clustering on images, signals, and time series: from data exploration and architecture design to training, robustness, uncertainty, and interpretation. Every group member must also be able to explain and reproduce the group's work in an oral exam.

## 2. Task Overview

Each group of 2 or 3 students completes three tasks: classification, regression, and clustering.

All groups receive the same data and the same questions: 100,000 images, 100,000 sensor signals and spectra, and the histories of 1,200 machines, plus `kamus_data.md`, which describes every file.

The test labels are hidden and used for automatic grading. The exam is a 2-week take-home followed by a 20-minute oral exam per group.

The total is 100 points: 85 points for group work (Tasks 1–3) and 15 points for the individual oral exam.

### Task 1 – Multimodal Classification: Product Defects (30 points)

A fictional component factory inspects every unit with a camera and a press sensor.
Classify each unit into seven classes: `normal`, `gores` (scratch), `noda` (stain), `retak` (crack), `gelembung` (bubble), `las_dingin` (cold weld), and `kontaminasi` (contamination).
Some defects show only in the image, some only in the sensor signal, and some are faint in both.

- **a. Exploration:** inspect examples of each class in both modalities, the class distribution, missing sensor recordings, and train–test differences (e.g. by production line).
- **b. Architecture:** design an image encoder (CNN or small ViT) and a signal encoder (1D-CNN, GRU, or Transformer). Compare at least two fusion strategies, e.g. early fusion, late fusion, or cross-attention.
- **c. Value of each modality:** train image-only and sensor-only models, and show which classes need which modality.
- **d. Imperfect labels:** training labels come from inspectors. Check for likely mislabels, then test one technique that is robust to label noise (e.g. label smoothing or co-teaching).
- **e. Domain shift:** measure performance per production line in validation, then test one technique to handle it (e.g. augmentation or per-line normalization).
- **f. Missing modality:** make sure the model still gives sensible predictions when the sensor was not recorded.
- **g. Training:** show a sanity check (overfitting one batch), learning curves, and results over at least three seeds for the final model.

**Scoring on the hidden test set:** macro-F1 over the seven classes.
**Points:** exploration and fusion architecture 10, modality, label-noise, and domain experiments 8, training and analysis 6, hidden test set 6 (macro-F1).

### Task 2 – Time-Series Regression: Remaining Useful Life (30 points)

The maintenance team wants to know how many more cycles each press machine can run before it fails.
Predict the remaining useful life (RUL) at the last recorded cycle of each test machine. RUL labels are capped at 150 cycles (anything above 150 counts as 150).

- **a. Exploration and preprocessing:** check how operating settings affect sensor readings, uninformative sensors, missing values, and spikes. Explain your normalization.
- **b. Labels:** build RUL labels for the training data from the run-to-failure histories, including the 150-cycle cap.
- **c. Sequences:** build sliding windows and handle histories of different lengths without leaking future information.
- **d. Models:** compare at least two sequence architectures (e.g. LSTM/GRU, 1D-CNN/TCN, or Transformer) with a non-deep-learning baseline.
- **e. Validation:** split by machine, not by row. Report RMSE and the asymmetric score per RUL range.
- **f. Uncertainty:** produce prediction intervals (e.g. MC dropout or an ensemble), and discuss when a machine should be serviced.

**Scoring on the hidden test set:** RMSE and an asymmetric score that penalizes late (over-optimistic) predictions more, with d = predicted RUL − actual RUL:

$$
s = \sum_{\text{machines}} \begin{cases} e^{-d/13} - 1 & d < 0 \\ e^{d/10} - 1 & d \ge 0 \end{cases}
$$

**Points:** preprocessing and labels 8, sequence models and baseline 8, validation and uncertainty 8, hidden test set 6 (RMSE: 3; asymmetric score: 3).

### Task 3 – Multimodal Clustering: Material Types (25 points)

A fictional laboratory has 36,000 unlabeled material samples, each with a microstructure image and a composition spectrum.
Group the samples by material type. The number of material types is not given.

- **a. Exploration:** inspect image distortions (rotation, scale, brightness, microscope differences) and spectrum distortions (shift, baseline, spikes).
- **b. Representation:** learn representations without labels (e.g. an autoencoder, contrastive learning such as SimCLR or BYOL, or pretrained embeddings). Explain the augmentations you use and why.
- **c. Two modalities:** build image and spectrum representations, then combine them. Show what is lost when only one modality is used.
- **d. Clustering:** compare at least two clustering methods on the embeddings, including one deep clustering approach such as DEC.
- **e. Number of clusters:** determine the number of material types with at least two criteria.
- **f. Analysis:** visualize the embeddings with UMAP or t-SNE, and explain which clusters are hard to separate and why.

**Scoring on the hidden reference:** Adjusted Rand Index (ARI) between your cluster labels and the true material types. Cluster numbers can be arbitrary.
**Points:** unsupervised representation 8, fusion of the two modalities 6, clusters and number of clusters 6, hidden reference 5 (ARI).

## 3. Submission Components

Submit through GitHub only. Each group keeps one repository, and its main branch must hold the six items below.
The prediction files are graded automatically, so file names and column names must match exactly.

- **a. `laporan.pdf`:** the report answering Tasks 1–3, at most 18 pages excluding appendices.
- **b. `kode/`:** all code plus a `README.md`. One command must reproduce every number in the report. Include the final model weights (or a download link for any file over 100 MB) or the steps to retrain them.
- **c. `prediksi_klasifikasi.csv`:** columns `id_unit`, `kelas`. One row per test unit; `kelas` is one of the seven class names exactly as written above.
- **d. `prediksi_rul.csv`:** columns `id_unit`, `rul`. One row per test machine; `rul` is between 0 and 150.
- **e. `cluster.csv`:** columns `id_sampel`, `cluster`. One row per sample; `cluster` is an integer.
- **f. `kontribusi.md`:** each member's role with links to their pull requests, and the group's use of AI tools, approved by all members.

Every member must fork the group repository, work in their own fork, and open pull requests to it. Contribution is assessed from GitHub (commits and pull requests).
Sync every fork and merge all pull requests before the deadline. The main branch at the deadline is what gets graded.

## 4. Submission Example

- **a. Repository layout (main branch):** `laporan.pdf`, `kode/` (with `README.md`), `prediksi_klasifikasi.csv`, `prediksi_rul.csv`, `cluster.csv`, `kontribusi.md`
- **b. Workflow:** one member creates the group repository, every member forks it and works in their fork, then opens pull requests that the group merges into main.
- **c. First lines of the prediction files (illustrative values):**

```
prediksi_klasifikasi.csv   id_unit,kelas      U123456,retak
prediksi_rul.csv           id_unit,rul        17,87.5
cluster.csv                id_sampel,cluster  S123456,4
```

## Deadline

The exam runs for 2 weeks from the data release.

1. Data release: **10 Nov 2026**
2. Submission deadline: **24 Nov 2026 23:59** (all forks synced and all pull requests merged)
3. Oral exam: TBD

## Notes

- **Methods:** the main model for every task must be deep learning. Classical methods are allowed as baselines.
- **Tools:** Python with PyTorch or TensorFlow/Keras. The free GPUs on Colab or Kaggle are enough.
- **Reproducibility:** every number in the report must be reproducible from your code with one command. State library versions and seeds.
- **External data:** not allowed. Publicly available pretrained weights may be used if you cite the source.
- **Collaboration:** sharing code, results, or findings between groups is forbidden. Because the data is the same, unusual similarity will be examined in the oral exam.
- **AI tools:** allowed. State the tool, the parts it helped with, and how the group verified the output.
- **Responsibility:** every member must be able to explain all of the group's work.
- **Late submission:** commits after the deadline count as late. The group score drops 10 points per 24 hours, and nothing is accepted after 48 hours.
- **Oral exam:** 5 minutes of reproduction (one member chosen by the lecturer runs the final model and reproduces a number from the report live), then 15 minutes of questions on any task, answered by members the lecturer chooses.
- **Final score** = group score × contribution factor + individual oral score.
- **Contribution factor:** 1.00 if the contribution shows on GitHub and the member can explain their claimed parts; 0.85 if the contribution exists but the explanation is weak; and 0.70 if the member cannot explain their claimed parts or contributed little.
- All work must be original and adhere to academic integrity standards.

## Links

- Dataset (`data_uts_dl.zip`): https://drive.google.com/drive/folders/15TiPGfsw0Irp36QM_zwd_oZnfcQquQ9x?usp=sharing
