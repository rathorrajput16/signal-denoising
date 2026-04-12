# ECG Denoising with OMP and CNN-LISTA

A biomedical ECG denoising pipeline built around sparse dictionary learning, with a classical **OMP** baseline and a learned **CNN-LISTA** model for fast, morphology-preserving reconstruction.

The pipeline uses **64-sample ECG patches**, a **128-atom K-SVD dictionary**, and **3 unrolled ISTA iterations** in the CNN-LISTA branch. The evaluation setup uses the **NSTDB dataset** and includes results for **patient 16265** (`nsrdb_16265e00.mat`).

---

## Overview

The goal of this project is to recover clean ECG morphology from heavily corrupted signals while keeping the computation efficient enough for real-time use. The classical sparse-coding path uses **Orthogonal Matching Pursuit (OMP)**, while the proposed method replaces the iterative optimizer with a **deep unrolled CNN-LISTA** network that learns the sparse inference process directly.

The result is a pipeline that keeps the interpretability of dictionary-based denoising and gains the speed of neural inference.

---

## Why Sparse Denoising?

ECG signals have structured morphology, especially around the QRS complex. Noise can hide or distort these patterns, and ordinary filtering often struggles when noise overlaps with the signal spectrum. Sparse dictionary learning models ECG patches as a combination of a small number of meaningful atoms, which makes the reconstruction both compact and clinically more faithful.

---

## Repository Structure

```text
.
├── ECG_data/
├── ecg_denoising/
│   ├── __init__.py
│   ├── baselines.py
│   ├── config.py
│   ├── data_loader.py
│   ├── dictionary.py
│   ├── metrics.py
│   ├── noise.py
│   └── visualization.py
├── inference_output/
│   ├── denoised_signal.npy
│   └── inference_result.png
├── inference_plots/
│   ├── denoised_101.png
│   └── denoised_200.png
├── models/
│   ├── ecg_dictionary.pkl
│   └── nstdb_dictionary.pkl
├── nstdb_denoising/
│   ├── __init__.py
│   ├── baselines.py
│   ├── config.py
│   ├── data_loader.py
│   ├── dictionary.py
│   ├── metrics.py
│   └── visualization.py
├── nstdb_plots/
├── plots/
├── .gitignore
├── README.md
├── inference.py
├── nstdb_inference.py
├── nstdb_train.py
└── train.py
```

---

## Pipeline 1: OMP Denoising

OMP is the classical sparse recovery baseline used in this project.

### Procedure

1. Start with the noisy ECG patch `y`.
2. Initialize the residual as `r = y`.
3. Select the dictionary atom most correlated with the residual.
4. Solve a least-squares problem on the selected atom set.
5. Update the residual.
6. Repeat for `k` iterations or until convergence.

### Mathematical Formulation

**Atom selection**

```text
λ_t = argmax |Dᵀ r_(t-1)|
```

**Coefficient update**

```text
α_t = (D_Λᵀ D_Λ)^(-1) D_Λᵀ y
```

**Residual update**

```text
r_t = y - D_Λ α_t
```

OMP is effective because it is morphology-aware, but it is also greedy, sequential, and expensive. Its complexity is approximately:

```text
O(k · M · N)
```

where:

- `k` = sparsity level
- `M` = number of dictionary atoms
- `N` = patch dimension

In practice, OMP becomes a CPU bottleneck because each step depends on the previous residual and cannot be fully parallelized. In extreme noise, it may also lock onto noise-like atoms instead of true cardiac structure.

---

## Pipeline 2: CNN-LISTA Denoising

CNN-LISTA is the learned version of the sparse-coding pipeline. It keeps the mathematical idea of ISTA but turns the iterative process into a trainable neural network.

CNN-LISTA replaces the classical Orthogonal Matching Pursuit stage with a deep unrolled model that maps noisy 64-sample patches to clean patches through **3 trainable iterations**. The model uses a **Conv1D encoder**, a **trainable soft-threshold**, and a **Dense decoder** initialized from a K-SVD dictionary.

### Procedure

1. Break the long ECG signal into 64-sample windows.
2. Subtract the patch mean to enforce zero-mean processing.
3. Encode the noisy patch using a Conv1D layer.
4. Apply soft-thresholding to obtain sparse coefficients.
5. Compute the reconstruction residual.
6. Repeat the refinement step for 3 unrolled iterations.
7. Decode the final sparse code with the learned dictionary.
8. Restore the reconstructed patch and discard the mean to avoid baseline drift.

### Mathematical Formulation

**Sparse coding objective**

```text
min over α of  (1/2) ||y - Dα||₂² + λ ||α||₁
```

**Classical ISTA update**

```text
α^(t+1) = S_λ( α^(t) + (1/L) · Dᵀ · (y - Dα^(t)) )
```

**CNN-LISTA update**

```text
α^(t+1) = S_λt( α^(t) + Wg · (y - Dα^(t)) )
```

**Soft-threshold operator**

```text
S_λ(z) = sign(z) · max(|z| - λ, 0)
```

### Loss Function

The model is trained end-to-end with a combined reconstruction and sparsity objective:

```text
L = (1/N) Σ ||y_true - y_recon||₂² + 0.001 · ||α||₁
```

### Key Components

- **Encoder (`W_e`)**: A Conv1D layer that maps the 64-sample signal into sparse feature space.
- **Threshold (`λ_t`)**: A trainable scalar at each iteration, so the model does not depend on hand-tuned sparsity thresholds.
- **Decoder (`D`)**: A Dense layer initialized from a **K-SVD dictionary** with **128 atoms × 64 samples** and made trainable during learning. Fine-tuning this decoder is critical for preventing collapse and improving recovery quality.

---

## Complete Mathematical Formulation

### 1. Sparse Representation

```text
y ≈ Dα
```

- `y`: noisy ECG patch
- `D`: dictionary
- `α`: sparse coefficient vector

### 2. Lasso Objective

```text
min over α of (1/2) ||y - Dα||₂² + λ ||α||₁
```

This encourages the patch to be explained by only a few dictionary atoms.

### 3. OMP Atom Selection

```text
λ_t = argmax |Dᵀ r_(t-1)|
```

The atom with the highest correlation to the current residual is selected at each step.

### 4. OMP Least-Squares Update

```text
α_t = (D_Λᵀ D_Λ)^(-1) D_Λᵀ y
```

The coefficients are recomputed using the selected atom subset.

### 5. OMP Residual Update

```text
r_t = y - D_Λ α_t
```

The residual is reduced by subtracting the reconstructed component.

### 6. ISTA Update

```text
α^(t+1) = S_λ( α^(t) + (1/L) · Dᵀ · (y - Dα^(t)) )
```

### 7. CNN-LISTA Update

```text
α^(t+1) = S_λt( α^(t) + Wg · (y - Dα^(t)) )
```

### 8. Soft Thresholding

```text
S_λ(z) = sign(z) · max(|z| - λ, 0)
```

### 9. Training Loss

```text
L = (1/N) Σ ||y_true - y_recon||₂² + 0.001 · ||α||₁
```

---

## Error Metrics

### Mean Squared Error (MSE)

```text
MSE = (1/N) Σ (y_i - ŷ_i)²
```

### Root Mean Squared Error (RMSE)

```text
RMSE = sqrt( (1/N) Σ (y_i - ŷ_i)² )
```

### Percent Root Difference (PRD)

```text
PRD = sqrt( Σ (y - ŷ)² / Σ y² ) × 100
```

### Coefficient of Determination (R²)

```text
R² = 1 - Σ (y - ŷ)² / Σ (y - ȳ)²
```

---

## Expressive Interpretation of Errors

A good ECG denoiser should not only reduce numbers on paper. It should preserve the clinical shape of the heartbeat.

When reconstruction fails, the error may appear as:

- a flattened QRS complex,
- a noisy residual spike left in the waveform,
- a drifting baseline that shifts the signal away from zero,
- or a visually smooth output that still loses the true cardiac morphology.

That is why the project evaluates both numerical metrics and waveform fidelity.

---

## Procedure Summary

### OMP Path

1. Load the noisy patch.
2. Initialize the residual.
3. Select the strongest matching atom.
4. Solve the projection step.
5. Update the residual.
6. Repeat until the sparsity limit is reached.
7. Reconstruct the denoised patch.

### CNN-LISTA Path

1. Load the noisy patch.
2. Subtract the patch mean.
3. Encode with Conv1D.
4. Apply learned thresholding.
5. Compute the residual.
6. Refine through 3 unrolled iterations.
7. Decode using the K-SVD-initialized dictionary.
8. Reconstruct the final clean patch.

---

## Results

On the **NSTDB dataset** using **patient 16265**, the learned CNN-LISTA model achieved strong denoising performance, including **9.83 dB output SNR**, **R² = 0.90**, **RMSE = 0.32**, and **PRD = 32.23%**. The model also reported **+10.31 dB improvement at -6 dB input noise** and was about **2.5× better than OMP** in that extreme-noise regime.

---

## Why CNN-LISTA Beats OMP

OMP is a strong baseline because it is interpretable and sparse. The weakness is that it is greedy and sequential, so performance drops when the residual is dominated by noise. CNN-LISTA improves on this by learning the update rule itself, keeping the sparse prior but making the inference path trainable, fixed-depth, and GPU-friendly. That is why its inference can scale to real-time use while still preserving ECG morphology.

---

## Complexity Comparison

### OMP

```text
O(k · M · N)
```

- sequential
- CPU-bound
- hard to vectorize
- slower for real-time deployment

### CNN-LISTA

```text
O(1)
```

- fixed 3 unrolled iterations
- GPU-parallelizable
- suitable for wearable deployment
- faster inference with comparable or better denoising quality

---

## Installation

```bash
git clone <repo-url>
cd <repo-name>
pip install -r requirements.txt
```

---

## Usage

### Train the model

```bash
python train.py
```

### Train on NSTDB

```bash
python nstdb_train.py
```

### Run inference

```bash
python inference.py
```

### Run NSTDB inference

```bash
python nstdb_inference.py
```

---

## Output Files

Typical generated outputs include:

- `inference_output/denoised_signal.npy`
- `inference_output/inference_result.png`
- plots in `inference_plots/`
- figures in `plots/` and `nstdb_plots/`

---

## Notes

- The dictionary is initialized using K-SVD.
- The decoder is trainable, not frozen.
- The model works on 64-sample ECG patches.
- Zero-mean patch processing helps prevent baseline drift.
- The repository includes both classical and learned denoising paths.

---

## Future Improvements

Possible extensions include:

- multi-lead ECG support
- more baseline methods
- better experiment dashboards
- TensorFlow Lite / ONNX export
- edge-device deployment

---

## License

Add your preferred license here.

---

## Acknowledgment

This project is built for ECG denoising with sparse dictionary learning and unrolled neural inference.
