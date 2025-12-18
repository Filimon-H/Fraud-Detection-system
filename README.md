# Fraud Detection System

A machine learning system for detecting fraudulent transactions in e-commerce and bank credit card data.

## Project Overview

This project builds fraud detection models that address the unique challenges of highly imbalanced transaction data. It includes:

- **Data Analysis & Preprocessing**: Cleaning, EDA, and geolocation enrichment
- **Feature Engineering**: Time-based features, transaction velocity, and geographic patterns
- **Model Building**: Baseline (Logistic Regression) and ensemble models (Random Forest, XGBoost)
- **Model Explainability**: SHAP analysis for interpretable predictions

## Project Structure

```
fraud-detection/
├── .github/workflows/      # CI/CD configuration
├── data/
│   ├── raw/               # Original datasets (gitignored)
│   └── processed/         # Cleaned and feature-engineered data
├── notebooks/
│   ├── 01_eda_fraud_data.ipynb
│   ├── 02_geo_ip_country.ipynb
│   ├── 03_feature_engineering_fraud.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_shap_explainability.ipynb
├── src/
│   ├── data/              # Data loading, cleaning, geolocation
│   ├── features/          # Feature engineering modules
│   ├── modeling/          # Model training and evaluation
│   └── utils/             # Utility functions
├── tests/                 # Unit tests
├── models/                # Saved model artifacts
├── scripts/               # Utility scripts
├── requirements.txt       # Python dependencies
└── README.md
```

## Datasets

### 1. Fraud_Data.csv (E-commerce)
E-commerce transaction data with features:
- `user_id`, `device_id`: Identifiers
- `signup_time`, `purchase_time`: Timestamps
- `purchase_value`, `age`: Numeric features
- `source`, `browser`, `sex`: Categorical features
- `ip_address`: For geolocation enrichment
- `class`: Target (1 = fraud, 0 = legitimate)

### 2. IpAddress_to_Country.csv
IP address range to country mapping for geolocation enrichment.

### 3. creditcard.csv (Bank)
Bank credit card transactions with anonymized PCA features (V1-V28).

## Setup

### 1. Create Virtual Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Add Data Files
Place the raw CSV files in `data/raw/`:
- `Fraud_Data.csv`
- `IpAddress_to_Country.csv`
- `creditcard.csv`

### 4. Run Notebooks
Execute notebooks in order:
1. `01_eda_fraud_data.ipynb` - Data exploration and cleaning
2. `02_geo_ip_country.ipynb` - IP to country mapping
3. `03_feature_engineering_fraud.ipynb` - Feature creation
4. `04_modeling.ipynb` - Model training (Task 2)
5. `05_shap_explainability.ipynb` - Model interpretation (Task 3)

## Key Features Engineered

### Time Features
- `hour_of_day`: Hour of purchase (0-23)
- `day_of_week`: Day of week (0=Monday)
- `is_weekend`: Weekend flag
- `time_since_signup`: Seconds between signup and purchase

### Velocity Features
- `tx_count_user_id_1h`: Transactions by user in last 1 hour
- `tx_count_user_id_24h`: Transactions by user in last 24 hours
- `user_total_transactions`: Total transactions per user

### Geographic Features
- `country`: Derived from IP address range lookup

## Handling Class Imbalance

The fraud datasets are highly imbalanced (~1-10% fraud rate). We address this by:
- Using **SMOTE** for oversampling the minority class (training only)
- Evaluating with **AUC-PR**, **F1-Score** instead of accuracy
- Stratified train-test splits to preserve class distribution

## Evaluation Metrics

- **AUC-PR (Precision-Recall AUC)**: Primary metric for imbalanced data
- **F1-Score**: Balance between precision and recall
- **Confusion Matrix**: Detailed breakdown of predictions
- **ROC-AUC**: Additional performance measure

## License

This project is for educational purposes (10 Academy Week 5-6 Challenge).
