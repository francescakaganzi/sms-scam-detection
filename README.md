# Fine-tuning LLMs for Mobile Money SMS Scam Detection

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v1.9+-red.svg)
![Transformers](https://img.shields.io/badge/🤗%20Transformers-v4.20+-yellow.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A project to asssess the effectiveness of fine-tuning LLMs for Mobile Money SMS scams dettections using traditional ML and deep learning as baselines, and large language models. This research compares multiple approaches from classical algorithms to state-of-the-art transformer models.

## 📊 Project Overview

This project implements and compares different approaches for Mobile Money SMS scam detection:

- **Traditional ML**: Logistic Regression, Random Forest, Decision Tree, XGBoost
- **Deep Learning**: TextCNN, BiLSTM
- **Transformer Models**: RoBERTa, DistilBERT (with Full, Frozen, and LoRA fine-tuning)
- **Large Language Models**: Llama-2 7B (with LoRA fine-tuning)

### Key Results

| Model | MCC | F1 Score | ROC-AUC | Efficiency |
|-------|-----|----------|---------|------------|
| **DistilBERT Full** | **0.861** | **0.875** | **0.952** | Medium |
| XGBoost | 0.815 | 0.832 | 0.923 | **High** |
| RoBERTa LoRA | 0.847 | 0.863 | 0.941 | Medium |
| BiLSTM | 0.798 | 0.821 | 0.915 | Medium |

> **Best Performance**: DistilBERT Full fine-tuning (MCC: 0.861)  
> **Most Efficient**: XGBoost (MCC: 0.815, fastest training)

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+
pip install torch torchvision torchaudio
pip install transformers datasets
pip install scikit-learn xgboost
pip install pandas numpy matplotlib seaborn
pip install nltk wordcloud
pip install optuna  # For hyperparameter optimization
pip install peft    # For LoRA fine-tuning
pip install bitsandbytes  # For quantization
```

### Google Colab Setup

```python
# Mount Google Drive
from google.colab import drive
drive.mount('/content/drive')

# Install required packages
!pip install transformers datasets peft bitsandbytes optuna
```

### Dataset

- **Size**: 1,000 SMS messages
- **Distribution**: 80% legitimate, 20% scam
- **Format**: CSV with `message` and `label` columns
- **Splits**: 70% train, 15% validation, 15% test

## 📁 Project Structure

```
sms-scam-detection/
├── data/
│   ├── raw/
│   │   └── spam-fraud-sms-dataset.csv
│   └── processed/
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── notebooks/
│   ├── 1_data_exploration.ipynb
│   ├── 2_baseline_ml_models.ipynb
│   ├── 3_deep_learning_models.ipynb
│   ├── 4_llm_finetuning.ipynb
│   └── 5_llama_finetuning.ipynb
├── models/
│   ├── baseline/
│   ├── deep_learning/
│   ├── llm/
│   └── llm_os/
├── results/
│   ├── metrics/
│   └── visualizations/
└── README.md
```

## 🔬 Methodology

### 1. Data Exploration and Preprocessing
- **Text Cleaning**: URL/email/phone tokenization, emoji removal, normalization
- **Feature Engineering**: Text length, word count, character ratios, pattern detection
- **Visualization**: Class distribution, word clouds, statistical analysis

### 2. Traditional Machine Learning
- **Feature Extraction**: TF-IDF vectorization (max_features=5000, ngrams=1-2)
- **Class Imbalance**: SMOTE oversampling
- **Hyperparameter Optimization**: Optuna with MCC as objective
- **Models**: Logistic Regression, Random Forest, Decision Tree, XGBoost

### 3. Deep Learning
- **Architecture**: TextCNN and BiLSTM with attention mechanisms
- **Tokenization**: Custom vocabulary (10k words, max_length=100)
- **Class Imbalance**: Weighted loss functions + balanced sampling
- **Regularization**: Dropout, batch normalization, early stopping

### 4. Transformer Fine-tuning
- **Models**: RoBERTa-base, DistilBERT-base-uncased
- **Approaches**: 
  - **Full Fine-tuning**: All parameters trainable
  - **Frozen**: Only classifier head trainable
  - **LoRA**: Parameter-efficient fine-tuning (r=16, α=32)
- **Optimization**: Adam optimizer, cosine scheduler, gradient clipping

### 5. Large Language Models
- **Model**: Llama-2 7B
- **Efficiency**: 4-bit quantization + LoRA fine-tuning
- **Memory**: Gradient checkpointing, mixed precision training

## 🏃‍♂️ Running the Code

### Option 1: Run All Experiments
```bash
# 1. Data exploration and preprocessing
jupyter notebook notebooks/1_data_exploration.ipynb

# 2. Traditional ML models
jupyter notebook notebooks/2_baseline_ml_models.ipynb

# 3. Deep learning models
jupyter notebook notebooks/3_deep_learning_models.ipynb

# 4. Transformer fine-tuning
jupyter notebook notebooks/4_llm_finetuning.ipynb

# 5. Llama fine-tuning (requires GPU)
jupyter notebook notebooks/5_llama_finetuning.ipynb
```

### Option 2: Google Colab (Recommended)
1. Upload notebooks to Google Colab
2. Mount Google Drive
3. Run notebooks in order
4. Results automatically saved to Drive

### Option 3: Quick Test
```python
# Load best model for quick predictions
from transformers import AutoTokenizer, AutoModelForSequenceClassification

model_name = "results/distilbert_full/final_model"
tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Test message
text = "Congratulations! You've won ugx100000. Click here to claim."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = model(**inputs)
prediction = torch.argmax(outputs.logits, dim=1).item()
print(f"Prediction: {'Scam' if prediction == 1 else 'Legitimate'}")
```

## 📈 Results and Analysis

### Performance Metrics

All models evaluated using:
- **Primary**: Matthews Correlation Coefficient (MCC)
- **Secondary**: F1-Score, Precision, Recall, Accuracy
- **Additional**: ROC-AUC, PR-AUC

### Key Findings

1. **Transformer Superiority**: DistilBERT achieved best performance (MCC: 0.861)
2. **Efficiency Trade-off**: XGBoost offers excellent performance/speed ratio
3. **LoRA Effectiveness**: 99.5% parameter reduction with minimal performance loss
4. **Class Imbalance**: Weighted loss functions crucial for imbalanced data

### Model Comparison

![Model Comparison](results/visualizations/final_all_models_comparison.png)

## 🛠️ Technical Details

### Hardware Requirements
- **Minimum**: 8GB RAM, CPU-only training
- **Recommended**: 16GB RAM + GPU (RTX 3070 or better)
- **For Llama**: 24GB+ GPU memory (or quantization)

### Training Times (on RTX 3070)
- **Traditional ML**: 2-5 minutes
- **Deep Learning**: 10-15 minutes
- **DistilBERT**: 8-12 minutes
- **RoBERTa**: 15-20 minutes
- **Llama-2 7B**: 45-60 minutes (with quantization)

### Memory Usage
- **Traditional ML**: <1GB
- **Deep Learning**: 2-4GB
- **DistilBERT**: 4-6GB
- **Llama-2 7B**: 14GB (4-bit quantized)

## 🔧 Configuration

### Key Hyperparameters

```python
# Traditional ML
{
    "tfidf_max_features": 5000,
    "ngram_range": (1, 2),
    "test_size": 0.15,
    "validation_size": 0.176
}

# Deep Learning
{
    "vocab_size": 10000,
    "max_seq_length": 100,
    "embedding_dim": 100,
    "batch_size": 32,
    "learning_rate": 1e-3
}

# Transformers
{
    "max_length": 128,
    "batch_size": 16,
    "learning_rate": 2e-5,
    "num_epochs": 3,
    "warmup_steps": 100
}

# LoRA Configuration
{
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["q_proj", "v_proj"]
}
```

## 📊 Reproducing Results

### For Exact Reproduction:
1. Use provided random seeds (`RANDOM_SEED = 42`)
2. Install exact package versions (see `requirements.txt`)
3. Use same data splits (saved in `data/processed/`)
4. Run notebooks in specified order

### Expected Results:
- DistilBERT Full: MCC ≥ 0.850
- XGBoost: MCC ≥ 0.800
- All models: F1 ≥ 0.800

## 🙏 Acknowledgments

- **Hugging Face** for transformer models and datasets library
- **Google Colab** for providing free GPU resources
- **Optuna** for hyperparameter optimization framework
- **scikit-learn** for traditional ML implementations

## 🔗 Related Work

- [Transformer Models for Text Classification](https://arxiv.org/abs/1810.04805)
- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [SMS Spam Detection: A Survey](https://link.springer.com/article/10.1007/s10462-021-09999-y)

## 📚 Additional Resources

- [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Google Colab Tips and Tricks](https://colab.research.google.com/)
- [Parameter-Efficient Fine-Tuning (PEFT)](https://github.com/huggingface/peft)
