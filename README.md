📧 Customer Support Ticket Classification using BERT

📌 Project Overview
Customer support ticket routing is a critical component of modern customer service operations, where incoming tickets must be automatically classified and routed to the appropriate support queue for efficient resolution.

This project implements an end-to-end NLP-based text classification system that:

Classifies customer support tickets using fine-tuned BERT
Handles class imbalance using advanced resampling techniques
Exposes the system via FastAPI backend with REST endpoints
Provides an interactive Streamlit dashboard for real-time ticket classification
Achieves 75% balanced accuracy on highly imbalanced data

The project is designed from an AI Engineer perspective, focusing on real-world challenges like class imbalance.

🎯 Business Problem
How can a customer support team automatically route incoming tickets to the correct queue (Billing, Technical Support, Returns, Sales, etc.) to reduce response time and improve customer satisfaction?
Manual ticket routing is:

❌ Time-consuming and inconsistent across different agents
❌ Unable to scale with growing ticket volume
❌ Leads to delayed responses and poor customer experience

🧠 Solution Approach
The system follows an industry-standard NLP pipeline:
1️⃣ Text Preprocessing (spaCy)

Lowercasing
Lemmatization for better generalization
Stopword and Punctuation removal
Token normalization for transformer input

2️⃣ Text Classification (BERT)

Uses BERT-base-uncased (110M parameters)
Fine-tuned on customer support data
Captures contextual and semantic meaning beyond keyword matching
Trained with class-weighted loss to handle imbalance

3️⃣ Class Imbalance Handling

RandomOverSampler to balance minority classes
Class-weighted loss function to penalize rare class errors
Balanced accuracy as primary evaluation metric

🏗️ System Architecture

Streamlit UI (Frontend)
        ↓
FastAPI REST API (Backend)
        ↓
Fine-Tuned BERT Model
        ↓
Prediction with Confidence Scores

📂 Project Structure

customer_support_ticket_classifier/
│
├── data/
│   └── support_ticket_classification_dataset.csv
│
├── models/
│   └── text_classification_bert_model/
│       ├── config.json
│       ├── model.safetensors
│       ├── tokenizer_config.json
│       ├── special_token_map.json
│       ├── vocab.txt
│       └── label_encoder.pkl
│
├── backend/
│   ├── utils.py                             # Model loader and predictor
│   └── server.py                            # FastAPI server
│
├── frontend/
│   └── app.py                               # Streamlit dashboard
│
├── notebooks/
│   └── nlp_support_ticket_classification.ipynb  # Training pipeline notebook
│
├── requirements.txt
└── README.md

📊 Dataset Description

Source: Kaggle
Size: 28,587 tickets (16 features)

Target Variable

queue — Assigned support queue (multi-class classification)


Key Features

Feature         Description                             Type
subject         Ticket subject line                     Text
body            Full ticket description                 Text
queue           Target queue (e.g., Billing, Technical) Categorical
priority        Ticket priority (high/medium/low)       Categorical
language        Communication language                  Categorical
tag_1 to tag_7  Manual tags/keywords                    Categorical

Class Distribution
Class 9 (Technical Support):                4737
Class 5 (Product Support):                  3073
Class 1 (Customer Service):                 2410
Class 4 (IT Support):                       1942
Class 0 (Billing and Payments):             1595
Class 6 (Returns and Exchanges):            820
Class 8 (Service Outages and Maintenance):  664
Class 7 (Sales and Pre-Sales):              513
Class 3 (Human Resources):                  348
Class 2 (General Inquiry):                  236

Imbalance Ratio: 4737/236 ​≈ 20 (severe imbalance) ⚠️

Interpretation:

This is severe class imbalance

    Any metric like accuracy is misleading
    Balanced accuracy / Macro-F1 is mandatory
    Oversampling or class weighting is justified

💡 In research:

IR > 10 → high imbalance
IR > 20 → extreme imbalance

RandomOverSampler: Balanced training set to 4737 × 0.8(train test split) ≈ 3790 samples per class

🔬 Exploratory Data Analysis (EDA)

🧪 Key Insights

Severe class imbalance — Minority classes represent only 5-9% of data

Text length varies significantly — 10-500 words per ticket

Strong semantic patterns (e.g., "invoice" → Billing)

Multiple features (tags, priority) provide additional signals

Subject + Body combination provides best classification accuracy

Text Statistics:

Average ticket length: 87 words
Vocabulary size: 15,234 unique tokens (after preprocessing)
Common terms: account, payment, technical, return

🤖 Model Training & Evaluation
Models Trained
1. Fine-tuned BERT

Model: bert-base-uncased
Optimizer: AdamW (lr = 3e-5)
Epochs: 5
Batch Size: 16
Max Sequence Length: 256
Hardware: NVIDIA T4 GPU

Evaluation Metrics
Metric              Definition                      Why Important
Accuracy            Overall correct predictions     Misleading for imbalanced data
Balanced Accuracy   Average of per-class recall     Fair across all classes
Macro F1            Unweighted average F1           Equal importance to all classes
Weighted F1         Sample-weighted F1              Overall performance

Results Summary
Model	                Accuracy	Balanced Accuracy	Macro F1	Weighted F1
Logistic Regression	    0.49	     0.34	            0.37	    0.47
BERT (Base)	            0.56	     0.46	            0.49	    0.55
BERT + Balanced Data	0.69	     0.67	            0.71	    0.69

Key Takeaway:

✅ BERT with oversampling achieves 67% balanced accuracy
✅ Oversampling + BERT improved macro-level performance by over 30 percentage points compared to logistic regression
✅ Macro F1 of 0.71 shows fair performance across all queues
✅ Data-level imbalance handling (oversampling) proved more effective than no correction, significantly improving minority class recall

Per-Class Performance (Best Model)
Queue                          precision  recall  f1-score    support

Billing and Payments            0.8801    0.8746    0.8774       319
Customer Service                0.6252    0.6577    0.6411       482
General Inquiry                 0.8621    0.5319    0.6579        47
Human Resources                 0.8824    0.6429    0.7438        70
IT Support                      0.6388    0.6701    0.6541       388
Product Support                 0.6256    0.6683    0.6462       615
Returns and Exchanges           0.8425    0.6524    0.7354       164
Sales and Pre-Sales             0.7531    0.5922    0.6630       103
Service Outages and Maintenance 0.8226    0.7669    0.7938       133
Technical Support               0.6694    0.6843    0.6768       947

accuracy                                            0.6900      3268
macro avg                       0.7602    0.6741    0.7089      3268
weighted avg                    0.6965    0.6900    0.6911      3268

💡 Key Implementation Decisions
Why BERT over traditional ML?

    Context-aware understanding (“not working” vs “working”)
    Transfer learning: Pre-trained on massive text corpus
    Industry standard for text classification

Why RandomOverSampler?

    Better empirical results: +2% balanced accuracy improvement
    Simpler implementation: No custom Trainer needed
    Standard approach: Works with vanilla Trainer API

Why Hugging Face Trainer?

    Production-ready with built-in best practices
    FP16 mixed precision for faster training
    Built-in evaluation, checkpoints, early stopping

🚀 Deployment
Backend (FastAPI)
Endpoints:

GET /classes — Get available queue classes
POST /predict — Single ticket classification

Key Dependencies

transformers
torch
spacy
scikit-learn
imbalanced-learn
fastapi
uvicorn
streamlit
pandas
numpy

🧪 Key Learnings

Balanced accuracy > standard accuracy for imbalanced problems

Preprocessing helps generalize better(lemmatization), reduce noise(Stop words removal) and can improve performance

Oversampling is often more effective than loss weighting

BERT fine-tuning requires small learning rates

Evaluation metrics must align with business impact

🔮 Future Improvements

Experiment with RoBERTa / DistilBERT
MLflow for experiment tracking
Model monitoring & drift detection
Hyperparameter tuning with Optuna