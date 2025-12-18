

10 Academy: Artificial Intelligence Mastery
Week 5&6 Challenge Document
Date: 17 Dec - 30 Dec 2025
Improved detection of fraud cases for e-commerce and bank transactions
Overview
Business Need
You are a data scientist at Adey Innovations Inc., a top company in the financial technology sector. Your company focuses on solutions for e-commerce and banking. Your task is to improve the detection of fraud cases for e-commerce transactions and bank credit transactions. 

This project aims to create accurate and strong fraud detection models that handle the unique challenges of both types of transaction data. It also includes using geolocation analysis and transaction pattern recognition to improve detection. Good fraud detection greatly improves transaction security. By using advanced machine learning models and detailed data analysis, Adey Innovations Inc. can spot fraudulent activities more accurately. This helps prevent financial losses and builds trust with customers and financial institutions.

A key challenge in fraud detection is managing the trade-off between security and user experience. False positives (incorrectly flagging legitimate transactions) can alienate customers, while false negatives (missing actual fraud) lead to direct financial loss. Your models should therefore be evaluated not just on overall accuracy, but on their ability to balance these competing costs. 

A well-designed fraud detection system also makes real-time monitoring and reporting more efficient, allowing businesses to act quickly and reduce risks.

This project will involve:

Analyzing and preprocessing transaction data.
Creating and engineering features that help identify fraud patterns.
Building and training machine learning models to detect fraud.
Evaluating model performance and making a justified selection.
Interpreting your model's decisions using modern explainability techniques.



Data and Features
You will be using the following datasets:
Fraud_Data.csv
	Includes e-commerce transaction data aimed at identifying fraudulent activities.
user_id: A unique identifier for the user who made the transaction.
signup_time: The timestamp when the user signed up.
purchase_time: The timestamp when the purchase was made.
purchase_value: The value of the purchase in dollars.
device_id: A unique identifier for the device used to make the transaction.
source: The source through which the user came to the site (e.g., SEO, Ads).
browser: The browser used to make the transaction (e.g., Chrome, Safari).
sex: The gender of the user (M for male, F for female).
age: The age of the user.
ip_address: The IP address from which the transaction was made.
class: The target variable where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.
Critical Challenge: Class Imbalance. This dataset is highly imbalanced, with far fewer fraudulent transactions than legitimate ones. This will significantly influence your choice of evaluation metrics and modeling techniques.
IpAddress_to_Country.csv
	Maps IP addresses to countries
lower_bound_ip_address: The lower bound of the IP address range.
upper_bound_ip_address: The upper bound of the IP address range.
country: The country corresponding to the IP address range.
creditcard.csv
Contains bank transaction data specifically curated for fraud detection analysis. 
Time: The number of seconds elapsed between this transaction and the first transaction in the dataset.
V1 to V28: These are anonymized features resulting from a PCA transformation. Their exact nature is not disclosed for privacy reasons, but they represent the underlying patterns in the data.
Amount: The transaction amount in dollars.
Class: The target variable, where 1 indicates a fraudulent transaction and 0 indicates a non-fraudulent transaction.
Critical Challenge: Class Imbalance. Like the e-commerce data, this dataset is extremely imbalanced, which is typical for fraud detection problems.
Learning Outcomes
Skills:
Effectively clean, preprocess, and merge complex datasets.
Engineer meaningful features from raw data.
Implement techniques to handle highly imbalanced datasets.
Train and evaluate models using metrics appropriate for imbalanced classification (e.g., AUC-PR, F1-Score).
Articulate and visualize model predictions using explainability tools like SHAP.
Knowledge:
Grasp the business and technical challenges of fraud detection.
Understand the importance of model explainability (XAI) for building trust and deriving insights.
Justify model selection based on both performance metrics and business context.
Behaviors:
Adopt a business-centric approach to problem-solving.
Demonstrate a systematic and organized workflow from data analysis to final interpretation.
Communication:
Reporting on statistically complex issues
Team
Tutors: 
Kerod 
Mahbubah
Filimon
Key Dates
Discussion on the case - 10:30 UTC on Wednesday, 17 Dec 2025.  
Interim-1 Submission - 20:00 UTC on Sunday, 21 Dec 2025.
Interim-2 Submission - 20:00 UTC on Sunday, 28 Dec 2025.
Final Submission - 20:00 UTC on Tuesday, 30 Dec 2025
Instructions
Project Structure
Organize your repository as follows:

fraud-detection/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/                           # Add this folder to .gitignore
│   ├── raw/                      # Original datasets
│   └── processed/         # Cleaned and feature-engineered data
├── notebooks/
│   ├── __init__.py
│   ├── eda-fraud-data.ipynb
│   ├── eda-creditcard.ipynb
│   ├── feature-engineering.ipynb
│   ├── modeling.ipynb
│   ├── shap-explainability.ipynb
│   └── README.md
├── src/
│   ├── __init__.py
├── tests/
│   ├── __init__.py
├── models/                      # Saved model artifacts
├── scripts/
│   ├── __init__.py
│   └── README.md
├── requirements.txt
├── README.md
└── .gitignore

Task 1 - Data Analysis and Preprocessing
Objective: Prepare clean, feature-rich datasets ready for modeling by exploring the data, engineering meaningful features, and handling class imbalance.

Instructions:

Data Cleaning
Handle missing values (impute or drop with justification)
Remove duplicates
Correct data types
Exploratory Data Analysis (EDA)
Univariate analysis: distributions of key variables
Bivariate analysis: relationships between features and target
Class distribution analysis: quantify the imbalance
Geolocation Integration
Convert IP addresses to integer format
Merge Fraud_Data.csv with IpAddress_to_Country.csv using range-based lookup
Analyze fraud patterns by country
Feature Engineering (for Fraud_Data.csv)
Transaction frequency and velocity: number of transactions per user in time windows.
Time-based features: 
hour_of_day, 
day_of_week
time_since_signup: duration between signup_time and purchase_time
Data Transformation
Normalize/scale numerical features (StandardScaler or MinMaxScaler)
Encode categorical features (One-Hot Encoding)
Handle Class Imbalance
Apply SMOTE or undersampling to training data only
Justify your choice of technique
Document the class distribution before and after resampling
Task 2 - Model Building and Training
Objective: Build, train, and evaluate classification models to detect fraudulent transactions, using appropriate techniques for imbalanced data.

Instructions:

Data Preparation
Split data using stratified train-test split (preserve class distribution)
Separate features from the target variable
creditcard.csv: 'Class'
Fraud_Data.csv: 'class'
Build Baseline Model
Train a Logistic Regression model as an interpretable baseline
Evaluate using AUC-PR, F1-Score, and Confusion Matrix
Build Ensemble Model
Train one of: Random Forest, XGBoost, or LightGBM
Perform basic hyperparameter tuning (e.g., n_estimators, max_depth)
Evaluate using the same metrics as baseline
Cross-Validation (recommended)
Use Stratified K-Fold (k=5) for reliable performance estimation
Report the mean and standard deviation of metrics across folds
Model Comparison and Selection
Compare all models side-by-side
Select the "best" model with a clear justification
Consider both performance metrics and interpretability
Task 3 - Model Explainability
Objective: Interpret your best model's predictions using SHAP to understand what drives fraud detection and provide actionable business recommendations.

Instructions:

Feature Importance Baseline
Extract built-in feature importance from your ensemble model
Visualize the top 10 most important features
SHAP Analysis
Generate SHAP Summary Plot (global feature importance)
Generate SHAP Force Plots for at least 3 individual predictions:
One true positive (correctly identified fraud)
One false positive (legitimate flagged as fraud)
One false negative (missed fraud)
Interpretation
Compare SHAP importance with built-in feature importance
Identify the top 5 drivers of fraud predictions
Explain any surprising or counterintuitive findings
Business Recommendations
Provide at least 3 actionable recommendations based on your analysis
Example: "Transactions within X hours of signup should receive additional verification."
Connect each recommendation to specific SHAP insights



Tutorials Schedule
Overview
In the following, the colour purple indicates morning sessions, and blue indicates afternoon sessions.
Wednesday
Introduction to the challenge (Kerod)
Fraud Detection Concepts(Filimon).
Thursday
Geolocation & IP Processing (Mahbubah)
Handling Imbalanced Data & Metrics for Imbalanced Classification (Smegnsh)
Friday
Model Building & Interpretation with SHAP(Filimon)
How to communicate insight from data (Mahbubah)
Monday
Q&A 
Tuesday
Q&A  
Thursday
Q&A
Monday 
Q&A
Deliverables 
Interim - 1 Submission Sunday 21 Dec, 2025
Focus: Confirm you have started your analysis and understand the datasets (Task 1).
What to Submit:
A link to your GitHub repository.
A detailed report including:
Summary of data cleaning and preprocessing steps
Key insights and visualizations from EDA
Explanation of feature engineering choices (especially time_since_signup and IP mapping)
Analysis of class imbalance and your strategy for handling it
Interim - 2 Submission Sunday 28 Dec, 2025
Focus: Confirm you have successfully built and evaluated at least one model (Task 2).
What to Submit:
A link to your GitHub repository.
Your repo should reflect the completion of the modeling task in addition to Interim-1
Feedback: You may not receive detailed comments on interim submissions, but you will receive a grade.
Final Submission Tuesday, 30 Dec 2025
Focus: A complete, well-documented end-to-end project.
What to Submit:
A link to your final GitHub repository that is professional and self-contained:
Comprehensive README.md with project overview and setup instructions
All code is organized into logical folders
Clear requirements.txt for environment setup
Blog Post or PDF Report narrating the entire project story:
Data analysis, feature engineering, and visualizations
Performance comparison of models with justification for the final choice
Screenshots and interpretation of SHAP plots
Feedback: You will receive detailed comments and feedback in addition to a grade.
References
Fraud Detection
Kaggle: Credit Card Fraud Dataset
Kaggle: IEEE Fraud Detection Competition
Kaggle: Fraud E-commerce Dataset
ComplyAdvantage: What is Fraud Detection?
Spiceworks: Fraud Detection Guide
Handling Imbalanced Data
imbalanced-learn Documentation
Analytics Vidhya: 10 Techniques to Handle Class Imbalance
Machine Learning Mastery: Tactics for Imbalanced Classes
SMOTE: Synthetic Minority Over-sampling Technique (Paper)
Evaluation Metrics
scikit-learn: Precision-Recall Curves
Machine Learning Mastery: ROC vs Precision-Recall Curves
Analytics Vidhya: Precision & Recall for Imbalanced Data
IP/Geolocation Processing
GeeksforGeeks: IP Address to Integer Conversion
pandas.merge_asof Documentation
MaxMind: Geolocation in Fraud Prevention
Modeling 
https://www.analyticsvidhya.com/blog/2021/08/conceptual-understanding-of-logistic-regression-for-data-science-beginners/
https://www.analyticsvidhya.com/blog/2021/08/decision-tree-algorithm/
https://www.analyticsvidhya.com/blog/2021/06/understanding-random-forest/
https://www.datacamp.com/tutorial/guide-to-the-gradient-boosting-algorithm
https://www.datacamp.com/tutorial/multilayer-perceptrons-in-machine-learning

Model Explainability
https://www.larksuite.com/en_us/topics/ai-glossary/model-explainability-in-ai
https://www.analyticsvidhya.com/blog/2021/11/model-explainability/
https://www.ibm.com/topics/explainable-ai
https://www.datacamp.com/tutorial/explainable-ai-understanding-and-trusting-machine-learning-models
