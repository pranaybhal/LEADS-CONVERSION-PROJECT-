Lead Scoring Analysis with Logistic Regression


Overview

This project aims to optimize the lead conversion process at X Education by building a machine learning model to predict potential leads (referred to as "Hot Leads") that are more likely to convert into paying customers. The project uses a logistic regression model and various data preprocessing techniques to improve the conversion rate.

The project leverages historical lead data, including information such as Lead Source, Total Time Spent on Website, Last Activity, and many other features. The end goal is to assign a lead score to each potential customer, helping the sales team prioritize their efforts towards more promising leads.

Table of Contents


Problem Statement
Project Objectives
Dataset Overview
Data Preprocessing
Model Building
Model Evaluation
Feature Importance
Installation
Usage
License


Problem Statement


X Education's lead conversion rate is only around 30%. To improve this, the company wants to identify the most potential leads, termed "Hot Leads." This project builds a model to assign a lead score to each lead, helping the sales team focus on high-probability conversions. The company aims to increase its lead conversion rate to approximately 80%.


Project Objectives


Preprocess the lead dataset by handling missing values, duplicates, and categorical data.
Build a logistic regression model to predict lead conversion.
Tune the model's hyperparameters using GridSearchCV.
Evaluate the model using accuracy, precision, recall, F1-score, and AUC-ROC curve.
Analyze feature importance to provide actionable insights for the sales team.
Provide business strategies based on model output for different scenarios, such as during high and low lead activity periods.


Dataset Overview
The dataset contains approximately 9000 rows, each representing a lead, with various attributes, including:

Lead Source: Where the lead originated (e.g., Google, Direct Traffic).
Total Time Spent on Website: How much time a lead spent on the website.
Last Activity: The last recorded interaction with the lead.
Converted: The target variable indicating whether the lead was converted (1 for yes, 0 for no).
For a complete list of features, refer to the Leads Data Dictionary.xlsx file in this repository.

Data Preprocessing

Several steps were taken to clean and prepare the dataset for modeling:

Handling Missing Values:
Replaced 'Select' values with NaN.
Imputed missing values for categorical variables using mode and for numerical variables using the median.
Removing Duplicates: Checked for and removed duplicate rows.
Feature Scaling: Scaled numerical features using StandardScaler.
Dummy Variables: Converted categorical variables into dummy variables using one-hot encoding.

Model Building

The model chosen for this project is Logistic Regression, which is suitable for binary classification tasks such as lead conversion (0/1).

Hyperparameter Tuning
GridSearchCV was used to find the best model parameters by tuning:

Regularization parameter C.
Solver (liblinear, lbfgs).

Model Evaluation

The model was evaluated using several key metrics:

Accuracy: 92%
Precision: 94%
Recall: 95%
F1-Score: 95%
AUC-ROC Score: 0.95

We also used a confusion matrix and ROC curve to visualize model performance. The high precision and recall values indicate that the model is good at identifying potential leads with minimal false positives and false negatives.

Feature Importance
The top features contributing to lead conversion were:

Total Time Spent on Website: Leads with more time on the website were more likely to convert.
Lead Source (Google): Leads coming from Google showed higher conversion probabilities.
Last Activity (Email Opened): Leads who opened emails showed strong conversion potential.
For more insights into feature importance, refer to the notebook or Python script in this repository.


Installation
Clone the repository:

git clone https://github.com/your_username/lead-scoring-analysis.git

Navigate to the project directory:

cd lead-scoring-analysis

Install the required Python packages:

pip install -r requirements.txt

Usage
Run the Jupyter notebook for the analysis step by step, or:
Run the Python script (if available):

python Leads.py
The model's output, including predictions and feature importance, will be displayed in the console or as plots.

License
This project is licensed under the MIT License. See the LICENSE file for more details.

