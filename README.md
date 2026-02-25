# Research and Application of Reinforcement Learning and Generative AI Algorithms for Image Data Augmentation

## Description

This project focuses on researching and applying Reinforcement Learning algorithms combined with Generative AI to enhance image data. The goal is to generate new data samples in the latent space, thereby improving the ability to detect anomalies and enhance the performance of computer vision models.

------------------------------------------------------------------------

## Datasets Used

-   **Official:**
    -   BMAD
-   **Additional (experimental):**
    -   PKU
    -   MV-Tec
    -   VisAD

------------------------------------------------------------------------

## Techniques Used

### Generative Model

-   Use AutoEncoder to learn latent representation `z` and reconstruct image data from this latent vector.

### Reinforcement Learning

-   Use PPO (Proximal Policy Optimization) algorithm to learn how to generate a noise `delta z`.
-   Create new latent vector:

```
z' = z + Δz
```

-   Vector `z'` is used to generate new image data through the decoder.

### Anomaly Detection

-   Use Transformer model to distinguish between normal and abnormal data.

------------------------------------------------------------------------

## Environment

-   Python 3.11
-   Libraries listed in `requirements.txt` file
-   Results and executed notebooks are saved in:

```
results-ipynb/
```

------------------------------------------------------------------------

## Running Instructions

### 1. Download Dataset

Download BMAD dataset from:

https://github.com/DorisBao/BMAD

------------------------------------------------------------------------

### 2. Data Preprocessing

-   Preprocess image data
-   Split data into:
    -   train
    -   validation
-   Save as `.npz` with:

```
allow_pickle=True
```

------------------------------------------------------------------------

### 3. Configure Path

Change data paths in `main` files.

------------------------------------------------------------------------

### 4. Train Model

Run:

    python main.py

------------------------------------------------------------------------

## Overall Pipeline

    Image → Encoder → z → PPO → Δz → z' → Decoder → Generated Image
                                          ↓
                                     Transformer
                                          ↓
                                  Anomaly Detection

