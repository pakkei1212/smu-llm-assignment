
# 📊 SMU LLM Assignment – Financial Sentiment Classification

## 📌 Project Overview

This project implements financial news sentiment classification using multiple transformer architectures and fine-tuning strategies.

The task is to classify financial headlines into:
- negative
- neutral
- positive

Both Full Fine-Tuning (FFT) and LoRA (Low-Rank Adaptation) are implemented and compared across multiple models.

---

## 📂 Repository Structure
```
smu-llm-assignment-main/
│
├── data/
│   └── all-data.csv
│
├── notebook/
│   ├── 00_testing.ipynb
│   ├── 01_data_loading.ipynb
│   ├── 01_data_loading-nvd.ipynb
│   ├── 02_training_lora(bert-base-cased).ipynb
│   ├── 02_training_lora(bert-base-uncased).ipynb
│   ├── 02_training_lora(deberta-v3-base).ipynb
│   ├── 02_training_lora(finbert-tone).ipynb
│   ├── 02_training_lora(qwen).ipynb
│   ├── 02_training_lora-nvd.ipynb
│   └── 03_evaluation.ipynb
│
├── src/
│   ├── BaseTrainer.py
│   ├── EncoderTrainer.py
│   ├── DecoderTrainer.py
│   ├── Seq2SeqTrainer.py
│   ├── ClassificationEvalTrainer.py
│   └── __init__.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
└── sbatchTemplatePython.sh
```
---

## 🧠 Architecture Design

### BaseTrainer.py
Abstract base class defining:
- Model loading
- Tokenizer setup
- Device resolution
- Training loop integration

### EncoderTrainer.py
Used for encoder-based models such as BERT, DeBERTa, and FinBERT.
Implements AutoModelForSequenceClassification for classification.

### DecoderTrainer.py
Used for causal LLMs (e.g., Qwen).
Handles instruction-style prompt formatting and generation-based classification.

### Seq2SeqTrainer.py
Used for sequence-to-sequence models (e.g., T5-style models).

### ClassificationEvalTrainer.py
Custom trainer extending HuggingFace Trainer:
- Custom loss computation (optional class weights)
- Macro F1 evaluation
- Extended metric logging

---

## 🔧 Fine-Tuning Methods

### 1️⃣ Full Fine-Tuning (FFT)
- Updates all model parameters
- Higher memory cost
- More expressive adaptation

### 2️⃣ LoRA
Implemented via PEFT:
- Injects low-rank matrices into attention layers
- Freezes backbone model
- Trains only a small percentage of parameters
- Reduces GPU memory usage significantly

---

## 📊 Dataset

File:
data/all-data.csv

Format:
label, text

Processing steps:
- Text cleaning
- Label normalization
- Stratified train/validation split

---

## 📏 Evaluation Metrics

Primary metric:
- Macro F1-score

Additional metrics:
- Accuracy
- Precision
- Recall

Macro F1 is used for:
- Early stopping
- Best model selection

---

## 🚀 How to Run

### Using Notebooks (Recommended)

1. Data loading:
   notebook/01_data_loading.ipynb

2. Training:
   notebook/02_training_lora(deberta-v3-base).ipynb

3. Evaluation:
   notebook/03_evaluation.ipynb

---

### Using Docker

docker-compose build
docker-compose up

---

## ⚙️ Environment Setup

pip install -r requirements.txt

Key libraries:
- transformers
- peft
- torch
- scikit-learn
- datasets

---

## 🎯 Experimental Focus

Hyperparameters explored:
- Learning rate
- LoRA rank (r)
- LoRA alpha
- LoRA dropout
- Class weights

Sequential progressive tuning strategy applied.

---

## 👤 Author

Patrick Yip  
Master of IT in Business (AI)  
Singapore Management University
