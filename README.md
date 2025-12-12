# Multi-Stage APT Attack Detection using Sequential Transfer Learning

This repository contains a Deep Learning framework designed to detect **Advanced Persistent Threats (APT)** by breaking the attack lifecycle into sequential stages. It utilizes **Sequential Transfer Learning** to train a chain of neural networks, passing "knowledge" from broad detection tasks to specific, high-precision classifications.

![Workflow Diagram](WORK_FLOW.png)

---

## Overview

Standard Intrusion Detection Systems (IDS) often struggle to differentiate between complex APT stages (e.g., *Exploit* vs. *Installation*). This framework solves that by using a **Divide-and-Conquer** approach through four main phases:

### 1. Data Mapping & Balancing
The system utilizes the **UNSW-NB15** dataset. It maps the original 9 attack categories into 5 logical APT stages:
- **Normal**: Benign traffic.
- **Reconnaissance**: Scanning, Analysis, Fuzzing.
- **Initial Compromise**: Exploits, Backdoors.
- **Exploit**: Denial of Service (DoS), Generic exploits.
- **Install**: Shellcode, Worms.
*Technique:* **SMOTE (Synthetic Minority Over-sampling Technique)** is applied to fix severe class imbalances.

### 2. Sequential Transfer Learning
Instead of training one giant model, we train a chain of specialized models. Weights from easier tasks are transferred to initialize harder tasks:
- **Stage 0**: Detect *Attack* vs *Normal*.
- **Stage 1**: Detect *Recon* vs *Others* (Transfer weights from Stage 0).
- **Stage 2**: Detect *Initial* vs *Others* (Transfer weights from Stage 1).
- **Stage 3**: Detect *Exploit* vs *Others* (Transfer weights from Stage 2).
- **Stage 4**: Detect *Install* vs *Others* (Transfer weights from Stage 3).

### 3. Refinement Layer
A specialized binary classifier is trained specifically to distinguish between **Exploit** and **Install** stages, which are statistically similar and often confused by standard models.

### 4. Cascade Inference Pipeline
During testing, samples are not classified largely. They pass through a **Decision Cascade**:
1.  Is it Malicious? (If No -> Normal)
2.  Is it Recon? (If Yes -> Recon)
3.  Is it Initial Access? (If Yes -> Initial)
4.  Refinement Check: Distinguish between *Exploit* and *Install* using the specialized model.

---

## Dependencies

All code is written in **Python 3**. The core logic is contained within a single **Jupyter Notebook**.

### Core Logic
- `pandas`: Data manipulation and mapping.
- `numpy`: Numerical operations.
- `scikit-learn`: Preprocessing (OneHotEncoder), Metrics, Class Weights.
- `imbalanced-learn`: SMOTE for data balancing.

### Deep Learning
- `tensorflow`: Core framework for Neural Networks.
- `keras`: High-level API for model building and transfer learning.

### Visualization
- `matplotlib` / `seaborn`: (Optional) For plotting confusion matrices.

---

## Description of Files

| File name | Description |
|:--- |:--- |
| `README.md` | Project documentation and methodology overview. |
| `requirements.txt` | List of Python dependencies required to run the notebook. |
| `UNSW_NB15_APT_features_train.csv` | Training dataset with extracted features from UNSW-NB15. |
| `UNSW_NB15_APT_features_test.csv` | Testing dataset used for final evaluation. |
| `APT_Detection_Pipeline.ipynb` | The main execution notebook. Handles data loading, SMOTE balancing, Sequential Transfer Learning (training loop), and the Cascade Inference logic. |

---

## Setup & Usage

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/AmritaCSN/Mrudul_APT_Detection.git]
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Prepare Data**
    Ensure `UNSW_NB15_APT_features_train.csv` and `test.csv` are in the root directory.

4.  **Launch Jupyter Notebook**
    ```bash
    jupyter notebook APT_Detection_Pipeline.ipynb
    ```
    *Note: Run cells sequentially. The training phase must complete (saving `.h5` files) before the Inference phase can run.*
