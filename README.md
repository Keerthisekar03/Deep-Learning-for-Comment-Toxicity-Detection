# 🛡️ Comment Toxicity Detection using Deep Learning & Streamlit

## 📌 Project Overview

This project is a **Deep Learning-based NLP application** designed to automatically detect toxic comments from user-generated text.

Toxic comments include:

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

The model is deployed using **Streamlit**, enabling real-time and bulk predictions.

---

## 🎯 Problem Statement

Online platforms face a major challenge in moderating harmful content such as hate speech, abusive language, and threats.

Manual moderation is:

* Time-consuming
* Not scalable
* Prone to human error

This project provides an **automated solution** to classify toxic comments efficiently.

---

## 🚀 Solution

We built a **multi-label text classification model** using Deep Learning that:

* Processes text input
* Predicts toxicity categories
* Displays confidence scores
* Supports bulk analysis via CSV

---

## 🧠 Tech Stack

* **Programming**: Python
* **Deep Learning**: TensorFlow / Keras
* **NLP**: TextVectorization
* **Frontend/UI**: Streamlit
* **Visualization**: Plotly
* **Data Handling**: Pandas, NumPy

---

## 📊 Dataset

* Multi-label dataset containing user comments
* Each comment is labeled across 6 toxicity categories
* Preprocessing includes:

  * Handling null values
  * Text normalization
  * Vectorization

---

## 🏗️ Model Architecture

* TextVectorization Layer
* Embedding Layer
* Bidirectional LSTM
* Global Max Pooling
* Dense Layers (Sigmoid Output)

---

## 🔄 Project Workflow

1. Data Collection & Cleaning
2. Text Preprocessing
3. Model Training (BiLSTM)
4. Model Evaluation
5. Model Saving (.h5)
6. Streamlit Deployment

---

## 💻 Application Features

* Real-time toxicity detection
* Confidence score visualization
* Adjustable threshold
* Bulk CSV upload
* Interactive UI

---

## ⚙️ Installation & Setup

### Step 1: Clone Repository

```bash
git clone https://github.com/your-username/toxicity-detector.git
cd toxicity-detector
```

### Step 2: Install Requirements

```bash
pip install -r requirements.txt
```

### Step 3: Run Application

```bash
python -m streamlit run app.py
```

---

## 📈 Example

### Input:

"You are useless and stupid"

### Output:

* Toxic: 88%
* Insult: 91%

---

## 📊 Business Use Cases

* Social media moderation
* Online forums filtering
* Brand safety monitoring
* E-learning platforms
* News website comment moderation

---

## 🔮 Future Improvements

* Implement BERT/Transformers
* Improve model accuracy
* Deploy to cloud (AWS/GCP)
* Multi-language support

---



