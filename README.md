# 🛡️ Comment Toxicity Detection

A deep learning NLP dashboard that detects toxic content in text comments using a **BiLSTM neural network** built with **PyTorch**, served via a **Streamlit** web app.

---

## 📌 Project Overview

This project classifies comments into 6 toxicity categories in real time:

| Category | Description |
|---|---|
| Toxic | General toxic language |
| Severe Toxic | Extremely harmful content |
| Obscene | Profane or vulgar language |
| Threat | Content threatening harm |
| Insult | Personal attacks |
| Identity Hate | Hate based on identity |

---

## 🧠 Model Architecture

```
Input Text
    ↓
Text Tokenization (custom vocabulary, 20,000 tokens)
    ↓
Embedding Layer (128 dimensions)
    ↓
Bidirectional LSTM (64 units × 2 directions = 128)
    ↓
Global Max Pooling
    ↓
Dense Layer (64 units, ReLU)
    ↓
Dropout (0.2)
    ↓
Output Layer (6 units, Sigmoid)
```

- **Framework:** PyTorch
- **Max sequence length:** 200 tokens
- **Vocabulary size:** 20,000
- **Training data:** Jigsaw Toxic Comment Classification dataset (159,571 comments)

---

## 📁 Project Structure

```
Comment_Toxicity/
│
├── app.py                        # Streamlit dashboard
├── toxicity_pytorch_model.pt     # Trained model weights
├── vectorizer_vocab.pkl          # Vocabulary file
├── train.csv                     # Training dataset
├── test.csv                      # Test dataset
└── README.md                     # This file
```

---

## ⚙️ Setup & Installation

### Requirements

- Python 3.10
- Windows / Linux / macOS

### Install dependencies

```bash
pip install torch streamlit pandas numpy plotly
```

### Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## 🚀 Features

### Single Comment Analysis
- Enter any comment and get instant toxicity scores for all 6 categories
- Visual bar chart with color-coded confidence scores
- Adjustable threshold slider (10%–90%)

### Bulk CSV Analysis
- Upload a CSV file with a `comment_text` column
- Analyze thousands of comments at once
- Download results as CSV

---

## 🏋️ Training (Google Colab)

The model was trained on Google Colab using a T4 GPU:

1. Load `train.csv` (Jigsaw dataset)
2. Build vocabulary from 159,571 comments
3. Tokenize and pad sequences to length 200
4. Train BiLSTM model for 3 epochs
5. Export weights as `toxicity_pytorch_model.pt`

Training time: ~15 minutes on T4 GPU

---

## 📊 Dataset

**Jigsaw Toxic Comment Classification Challenge**
- Source: [Kaggle](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)
- 159,571 training comments
- 6 binary labels per comment
- Multi-label classification (a comment can be multiple categories)

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| Deep Learning | PyTorch |
| Web Dashboard | Streamlit |
| Data Processing | Pandas, NumPy |
| Visualization | Plotly |
| Training Environment | Google Colab (T4 GPU) |
| IDE | VS Code |

---

## 📸 Screenshots

> screenshots - dashboard here after uploading.
> 
> If you predict comments like positive then type "Nice" and the output will be:
> 
> <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/19ea6484-d091-43f4-a904-5b590c0fc980" />

> If you predict comments like negative then type "Idiot"  and the output will be:

<img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/968f5990-3def-40df-af1c-2d8feb7d1f99" />


---

## 👤 Keerthika J S

Built from scratch as a deep learning NLP project.

---


