# Credit Card Fraud Detection

HI! This is a machine learning system that detects fraudulent credit card transactions using Random Forest and SMOTE oversampling to handle class imbalance.

## Results
| Metric | Score |
|--------|-------|
| Accuracy | 99.96% |
| Precision | 84.5% |
| Recall | 84% |
| F1 Score | 84% |

## Tech Stack
- Python
- scikit-learn (Random Forest)
- imbalanced-learn (SMOTE)
- pandas
- pytest

## How to Run
1. Clone this repo
2. Install dependencies:
   pip install -r requirements.txt
3. Download dataset from Kaggle and place creditcard.csv in data/ folder
   https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
4. Run tests:
   python -m pytest tests/ -v

## Project Structure
- src/preprocess.py — loads data, scales features, applies SMOTE
- src/trainmodel.py — trains Random Forest model
- src/evaluate.py — calculates accuracy, precision, recall, F1
- tests/test_model.py — automated pytest test suite

## Dataset
284,807 credit card transactions with only 0.17% fraud cases.
Dataset source: Kaggle (ULB Machine Learning Group)
