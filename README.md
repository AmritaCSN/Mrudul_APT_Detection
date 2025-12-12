# Multi-Stage APT Attack Detection using Sequential Transfer Learning

This repository contains a Deep Learning framework designed to detect **Advanced Persistent Threats (APT)** by breaking the attack lifecycle into sequential stages. It utilizes **Sequential Transfer Learning** to train a chain of neural networks, passing "knowledge" from broad detection tasks to specific, high-precision classifications.

![Workflow Diagram](https://mermaid.ink/img/pako:eNqVVE1v2zAM_SuEzmkD-4PtYeimFwiwYTsM3Q4FisJYZC2RPEiK6wb97yMlx06zYdsO80WSD4-Pj1R6NqTckDTm-WvOizXn8sOaF2u2YVzJteBawg-F5K_S8oK_ryUv2Iat1qwQgiXnF_yCl2z1dsmW6yVb8Tf2ki1X6yXjG37B1zdc8A3jv2u25Pztmq3YavWOv_4i5D8X6y3bvN-y1Q8uFbyScs0KqXm5ZqWQC8654G_s5Qe74K9fXvAFX7G1lEu-4a-s4Bs2F3zF31jJ1-yCr9nLNZfzFf-5YfPFki35y3v2csU3fM1W6xVf8Zc127DXa7Zib9Z8w97wF7ZabfgLXtJ5wzfs9XbJlnzN31jJNVvyF7Z6v2Fv-Cv-esNf2Iat3m74G3vD3vDXDVvyF7yUv5ZsyV-54K_v-OuaLdgb9nLDN/yVlWz1dsPe8De25Bu24W_sDXvDX97zF7Z6v2Fv-Bu75Bv-esNf2Iat3m74G3vD3vDXDVvyF7yUv5Zsydf8jZVsyV_Y6v2GveGv-OsNf2Ebtmq_8X9wQ7HkQshCciE4F2suJV-z5zV7xV9v-Qvf8C17vV2yJV_zN1byNbtgG77hrzd8w19ZyVZvN3zDX9mSr9mSv7DXG77hr6xk67cbvuGvrORr/sZKtvyf3fANf2UlW73d8A1/ZUu-Zkv-wl5v+Ia_spKt3274hr-ykq_5GyvZMv3XbMlf2OsN3_BXVrL12w3f8FdW8jV_YyVbpv-aLfkLe73hG_7KSrZ-u-Eb_spKvuZvrGTL9F-zJX9hrzd8w19Zyda03fANf2UlX_M3VrJl-q_Zkr-w1xu-4a-sZOu3G77hr6zka_7GSrZM_zVb8hf2esM3_ZWVbP12wzf8lZV8zd9YyZbpv2ZL_sJeb_iGv7KSrd9u-Ia_spKv-Rsr2TL912zJX9jrDd_wV1ay9dsN3_BXVvI1f2MlW6b_mi35C3u94Rv-ykq2frvhG_7KSr7mb6xky_RfsyV/Ya83fMNfWcnWbzd8w19Zydf8jZVsmf5rtuQv7PWGb_grK9n67YZv-CkrVkglaSkEkyXnZckrUfBKclFwKSUvhKyU4FwIqSStBC8FL0vBy1JwXgpeCo5zKQqpeCFKyf8Bl2_uXA)

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
    git clone [https://github.com/YourUsername/APT-Detection-Transfer-Learning.git](https://github.com/YourUsername/APT-Detection-Transfer-Learning.git)
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
