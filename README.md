# Fake Review Detection

## Overview
This project aims to detect fake reviews using machine learning techniques based on linguistic and behavioral features. The dataset consists of restaurant reviews labeled as real or fake. Features like average word/sentence length, part-of-speech tags, content diversity, and typos are used to classify the reviews.

We implemented and evaluated four models:

- Logistic Regression  
- Random Forest  
- XGBoost  
- Support Vector Machine (SVM)  

Performance was assessed using accuracy and macro F1-score. Preprocessing techniques such as text cleaning, tokenization, typo detection, and Recursive Feature Elimination (RFECV) were applied to enhance model performance.

---

## Files in the Repository

### 1. `Fake Review Detection.ipynb`
Contains all code for:

#### Dataset Preprocessing:
- Cleaning text (lowercase, remove punctuation/numbers, stopwords)
- Tokenization (words & sentences using `nltk`)
- Typo detection using `TextBlob`
- Content diversity calculation
- Dataset splitting (60% train, 20% validation, 20% test)
- Feature selection using RFECV with Random Forest

#### Feature Engineering:
- Average Word Length (AWL)
- Average Sentence Length (ASL)
- Number of Words (NWO)
- Number of Verbs (NVB)
- Number of Adjectives (NAJ)
- Number of Passive Voice Constructions (NPV)
- Number of Sentences (NST)
- Content Diversity (CDV)
- Number of Typos (NTP)
- Typo Ratio (TPR)

#### Model Training:
- Logistic Regression
- Random Forest
- XGBoost
- Support Vector Machine (SVM)

#### Evaluation & Visualization:
- Accuracy and Macro F1-score
- Confusion matrices
- Correlation heatmap
- Word clouds for real and fake reviews

---

### 2. `Fake Review Detection Report.pdf`
A detailed project report including:

- Importance of fake review detection in digital marketplaces
- Dataset characteristics and label descriptions
- Linguistic feature extraction techniques
- Preprocessing workflow
- Hyperparameter tuning strategies
- Comparative model performance
- Insights and future improvement directions

---

## Project Workflow

### Step 1: Dataset Preprocessing
- Text normalization and stopword removal
- Sentence and word tokenization
- Typo detection via `TextBlob`
- Feature extraction for linguistic and behavioral indicators
- Train/validation/test split (stratified)
- Feature selection with RFECV using Random Forest

### Step 2: Model Training
Train the following models:
- **Logistic Regression** (baseline classifier)
- **Random Forest** (ensemble decision trees)
- **XGBoost** (gradient boosting)
- **Support Vector Machine (SVM)** (max-margin classifier)

### Step 3: Hyperparameter Tuning
- `RandomizedSearchCV` with Stratified 5-Fold Cross Validation
- Tuning done on combined training and validation sets
- Best parameters selected based on F1-score

### Step 4: Evaluation
- Accuracy
- Macro F1-score
- Confusion matrix analysis

---

## Results Summary

| Model               | Train Acc | Train F1 | Test Acc | Test F1 |
|--------------------|-----------|----------|----------|---------|
| Logistic Regression| 67.3%     | 0.688    | 63.5%    | 0.641   |
| Random Forest      | 85.1%     | 0.841    | 67.3%    | 0.638   |
| XGBoost            | 99.0%     | 0.990    | 63.5%    | 0.612   |
| SVM                | 69.2%     | 0.677    | 65.4%    | 0.591   |

---

## Key Insights

- **Random Forest** provided the most reliable performance, balancing accuracy and generalization.
- **XGBoost** showed overfitting—great training results but poor test generalization.
- **Logistic Regression** generalized well with a small performance gap between training and test sets.
- **SVM** achieved high test accuracy but a lower F1-score, indicating class imbalance sensitivity.
- **Typo-related features** (NTP, TPR) were important for tree-based models in distinguishing fake reviews.

---
