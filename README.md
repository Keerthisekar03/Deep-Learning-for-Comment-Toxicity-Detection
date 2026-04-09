# 🛡️ Comment Toxicity Detection using Deep Learning & Streamlit

## 📌 Project Overview

This project is a **Deep Learning-based NLP application** designed to automatically detect toxic comments from user-generated text. 
This is addresses the challenge of toxic behavior in online communities by using Deep Learning to detect and flag harassment, hate speech, and offensive language in real-time. Built with Python and TensorFlow, the system classifies comments into six distinct toxicity categories, providing platform moderators with an automated tool for healthier discourse.

Toxic comments include:

* Toxic
* Severe Toxic
* Obscene
* Threat
* Insult
* Identity Hate

The model is deployed using **Streamlit**, enabling real-time and bulk predictions.

---
## 🚀 Features

- Multi-Label Classification: Detects six categories: Toxic, Severe Toxic, Obscene, Threat, Insult, and Identity Hate.

- Real-time Analysis: Interactive Streamlit web interface for instant text prediction.

- Bulk Processing: Supports CSV file uploads for large-scale comment analysis.

- Interactive Visualizations: Dynamic bar charts using Plotly to show confidence scores.

- Customizable Thresholds: Sidebar slider to adjust sensitivity for flagging comments.

## 🎯 Problem Statement

Online platforms face a major challenge in moderating harmful content such as hate speech, abusive language, and threats.

Manual moderation is:

* Time-consuming
* Not scalable
* Prone to human error

This project provides an **automated solution** to classify toxic comments efficiently.

---

## 🛠️ Technical Stack

Deep Learning Framework: TensorFlow/Keras.

Architecture: Bi-directional LSTM (Long Short-Term Memory) with an Embedding layer.

Frontend: Streamlit.

Data Handling: Pandas, NumPy.

Visualizations: Plotly.

---

## 🏗️ Project Workflow

Data Preprocessing: Handling missing values and standardizing text input.

Vectorization: Converting text into numerical sequences using a TextVectorization layer with a 20,000-word vocabulary.

Model Training: A Sequential model utilizing Bi-LSTM layers to understand context in both directions of a sentence.

Evaluation: Monitoring accuracy and loss over multiple epochs.

Deployment: Exporting the model as a .h5 or .keras file and integrating it into a Streamlit dashboard.

---

## 🚀 Solution

We built a **multi-label text classification model** using Deep Learning that:

* Processes text input
* Predicts toxicity categories
* Displays confidence scores
* Supports bulk analysis via CSV

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

* Social Media Platforms: Automated real-time filtering of toxic content.
* Online Forums: Assisting moderators in flagging high-risk threads.
* Brand safety monitoring
* Educational Platforms: Ensuring safe and constructive peer-to-peer communication.
* News website comment moderation

---

## 🔮 Future Improvements

* Implement BERT/Transformers
* Improve model accuracy
* Deploy to cloud (AWS/GCP)
* Multi-language support

---
# Domain: Online Community Management & Content Moderation



