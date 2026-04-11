# ECG Denoising with CNN-LISTA

A biomedical signal denoising pipeline for removing EM noise from ECG signals using a **dictionary-learning baseline** and a **Deep Unrolled CNN-LISTA** model.

---

## Overview

This project focuses on denoising noisy ECG patches using a sparse representation framework. The classical sparse-coding step is first built with a dictionary-learning baseline, then replaced with a trainable **CNN-LISTA** architecture that unrolls ISTA updates into a fixed number of neural layers.

The key idea is simple:

- represent ECG patches sparsely
- preserve cardiac morphology
- replace slow iterative optimization with fast learned inference

This makes the pipeline suitable for real-time and wearable ECG applications.

---

## Core Idea

For a noisy ECG patch \( y \), we approximate:

\[
y \approx D\alpha
\]

where:

- \( y \) is the noisy ECG patch
- \( D \) is the dictionary
- \( \alpha \) is the sparse coefficient vector

The sparse coding objective is:

\[
\min_{\alpha} \frac{1}{2}\|y - D\alpha\|_2^2 + \lambda \|\alpha\|_1
\]

This balances two things:

- **reconstruction accuracy**
- **sparsity**

The result is a clean reconstruction that keeps the important ECG structure, especially the QRS morphology.

---

## Why This Matters

EM noise can look deceptively similar to real ECG activity. That is what makes the problem difficult. A filter that removes too much can flatten the heartbeat. A filter that removes too little leaves the signal unreadable.

This project addresses that problem by using learned sparse reconstruction instead of relying only on classical frequency filtering.

---

## CNN-LISTA Update Rule

The classical ISTA iteration is:

\[
\alpha^{(t+1)} = S_{\lambda}\left(\alpha^{(t)} + \frac{1}{L}D^T(y - D\alpha^{(t)})\right)
\]

In the learned version, the update becomes:

\[
\alpha^{(t+1)} = S_{\lambda_t}\left(\alpha^{(t)} + W_g \cdot (y - D\alpha^{(t)})\right)
\]

where:

- \( W_g \) is trainable
- \( \lambda_t \) is learned per iteration
- the residual path is encoded using convolution
- the decoder is initialized from K-SVD and fine-tuned during training

The soft-threshold operator is:

\[
S_{\lambda}(z) = \text{sign}(z)\max(|z| - \lambda, 0)
\]

---

## Error Metrics

This project uses multiple error measures to show how well the denoiser preserves the ECG waveform.

### 1. Mean Squared Error (MSE)

\[
\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2
\]

MSE punishes large mistakes heavily. If a QRS peak is lost or badly distorted, MSE rises sharply.

### 2. Root Mean Squared Error (RMSE)

\[
\text{RMSE} = \sqrt{\frac{1}{N}\sum_{i=1}^{N}(y_i - \hat{y}_i)^2}
\]

RMSE is easier to interpret because it is in the same scale as the signal.

### 3. Percent Root Difference (PRD)

\[
\text{PRD} = \sqrt{\frac{\sum (y - \hat{y})^2}{\sum y^2}} \times 100
\]

PRD shows how much of the original waveform energy was lost in reconstruction.

### 4. Coefficient of Determination \( R^2 \)

\[
R^2 = 1 - \frac{\sum (y - \hat{y})^2}{\sum (y - \bar{y})^2}
\]

A higher \( R^2 \) means the reconstructed signal explains the original waveform better.

---

## Expressive Error Analysis

When denoising fails, it does not always fail quietly.

Sometimes the error appears as a **flattened QRS complex**, where the heartbeat loses its sharp shape.  
Sometimes it shows up as **residual noise spikes** that survive the reconstruction and make the ECG look jagged.  
Sometimes the signal suffers from **baseline drift**, where the waveform slowly floats up or down and stops feeling clinically reliable.  
And sometimes the reconstruction is technically smooth but still wrong in the worst way: it looks clean, but the heart’s true morphology is gone.

That is why this project does not rely on a single metric. It measures reconstruction error, sparsity, signal fidelity, and variance capture together.

---

## Key Results

From the evaluation shown in the presentation:

- **Output SNR:** 9.83 dB
- **Input SNR:** 4.43 dB
- **Improvement:** +5.40 dB
- **\( R^2 \):** 0.90
- **RMSE:** 0.32
- **PRD:** 32.23%

The model performs especially well at extreme noise levels, where classical methods begin to collapse.

---

## Project Structure

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
