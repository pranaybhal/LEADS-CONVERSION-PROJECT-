# %%
# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualizations
import seaborn as sns  # For advanced visualizations
from sklearn.model_selection import train_test_split  # For splitting data into training and testing
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve  # For evaluating model


# %%


# Load the CSV file using the full file path
file_path = r'C:\Users\Lenovo\Downloads\Lead+Scoring+Case+Study\Lead Scoring Assignment\Leads.csv'

# Load the data
df = pd.read_csv(file_path)

# Display the first few rows of the dataframe
df.head()



# %%
# Basic info about the dataset (columns, data types, missing values, etc.)
df.info()

# Get summary statistics for numerical columns
df.describe()

# Check for any missing values in each column
df.isnull().sum()


# %%
# Replace 'Select' with NaN as it is equivalent to missing data
df.replace('Select', pd.NA, inplace=True)

# Check again for missing values
df.isnull().sum()

# Handling missing values
# Option 1: Drop rows with missing values (can be harsh if too many rows are dropped)
df.dropna(inplace=True)

# Option 2: Impute missing values (for categorical and numerical columns)
# Impute missing values for categorical columns with mode (most frequent value)
categorical_columns = df.select_dtypes(include=['object']).columns
for col in categorical_columns:
    df[col].fillna(df[col].mode()[0], inplace=True)

# Impute missing values for numerical columns with median (robust to outliers)
numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
for col in numerical_columns:
    df[col].fillna(df[col].median(), inplace=True)

# Verify no missing values remain
df.isnull().sum()


# %%
# Check for duplicate rows
duplicates = df.duplicated().sum()

# If there are duplicates, remove them
if duplicates > 0:
    df.drop_duplicates(inplace=True)

# Confirm the removal of duplicates
df.duplicated().sum()


# %%
# Create dummy variables for categorical columns
df = pd.get_dummies(df, drop_first=True)

# Check the resulting dataframe after creating dummy variables
df.head()


# %%
# Check the data types of each column to ensure they are appropriate
df.dtypes

# Ensure no missing or duplicate values remain
df.isnull().sum()
df.duplicated().sum()

# Final dataset ready for analysis
df.head()


# %% [markdown]
# Loading the Data: You load the CSV file containing leads information using pandas.
# 
# Understanding the Data: You inspect the dataset to understand the columns, data types, and any initial missing values.
# 
# Handling Missing Values: Missing values are treated by either dropping rows or imputing based on the column type (mode for categorical, median for numerical).
# 
# Removing Duplicates: You check for any duplicate rows in the dataset and remove them.
# 
# Creating Dummy Variables: Categorical columns are converted to dummy variables, which are essential for modeling.
# 
# Ensuring Data Quality: You perform final checks on data types, missing values, and duplicates to ensure the data is ready for modeling.

# %%
# Define the features (X) and target variable (y)
X = df.drop('Converted', axis=1)  # Drop the target column from the feature set
y = df['Converted']  # Define the target column

# Split the data into 70% training and 30% testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shapes of the training and testing sets
print(X_train.shape, X_test.shape)


# %%
# Initialize the Logistic Regression model
logreg = LogisticRegression(solver='liblinear', random_state=42)

# Fit the model on the training data
logreg.fit(X_train, y_train)

# Predict on the testing data
y_pred = logreg.predict(X_test)

# Evaluate the model using accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")


# %%
# Import necessary libraries
import pandas as pd  # For data manipulation
import numpy as np  # For numerical operations
import matplotlib.pyplot as plt  # For visualizations
import seaborn as sns  # For advanced visualizations
from sklearn.model_selection import train_test_split, GridSearchCV  # For splitting data and grid search
from sklearn.preprocessing import StandardScaler  # For feature scaling
from sklearn.linear_model import LogisticRegression  # Logistic regression model
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve  # For evaluating model

# Load your dataset (Assuming the data has been preprocessed)
# df = pd.read_csv('your_preprocessed_data.csv')

# Define your features (X) and target variable (y)
X = df.drop('Converted', axis=1)  # Drop the target column
y = df['Converted']  # Define the target column

# Split the dataset into training and testing sets (70% training, 30% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features (scaling)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Initialize Logistic Regression model
logreg = LogisticRegression(random_state=42)

# Set up the hyperparameter grid for GridSearchCV
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs'],  # Different solvers for optimization
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5, scoring='accuracy')

# Fit the model on the training data
grid_search.fit(X_train, y_train)

# Get the best parameters and estimator
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Predict on the test data using the best model
y_pred = best_model.predict(X_test)

# Evaluate the best model's performance
accuracy = accuracy_score(y_test, y_pred)
print(f"Best Accuracy: {accuracy:.2f}")

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Classification report
class_report = classification_report(y_test, y_pred)
print("Classification Report:")
print(class_report)

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, best_model.predict_proba(X_test)[:, 1])
print(f"AUC-ROC Score: {roc_auc:.2f}")

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, best_model.predict_proba(X_test)[:, 1])
plt.figure()
plt.plot(fpr, tpr, label=f"ROC Curve (area = {roc_auc:.2f})")
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()


# %%
# Set up the hyperparameter grid for tuning
param_grid = {
    'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
    'solver': ['liblinear', 'lbfgs'],  # Solvers for optimization
}

# Use GridSearchCV to find the best parameters
grid_search = GridSearchCV(LogisticRegression(random_state=42), param_grid, cv=5, scoring='accuracy')

# Fit the model with the best parameters
grid_search.fit(X_train, y_train)

# Get the best parameters
print("Best parameters found: ", grid_search.best_params_)

# Predict on the test data using the best model
y_pred_best = grid_search.best_estimator_.predict(X_test)

# Evaluate the best model
accuracy_best = accuracy_score(y_test, y_pred_best)
print(f"Best Accuracy: {accuracy_best:.2f}")


# %%
# Import necessary metrics at the beginning of your code
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Evaluate the best model's performance

# Accuracy 
accuracy = accuracy_score(y_test, y_pred_best)
print(f"Accuracy: {accuracy:.2f}")

# Precision
precision = precision_score(y_test, y_pred_best)
print(f"Precision: {precision:.2f}")

# Recall
recall = recall_score(y_test, y_pred_best)
print(f"Recall: {recall:.2f}")

# F1-Score
f1 = f1_score(y_test, y_pred_best)
print(f"F1-Score: {f1:.2f}")

# AUC-ROC score
roc_auc = roc_auc_score(y_test, grid_search.best_estimator_.predict_proba(X_test)[:, 1])
print(f"AUC-ROC Score: {roc_auc:.2f}")


# %%
# Save the feature names before scaling
feature_names = X.columns  # This is the original column names of your data

# Get the coefficients of the logistic regression model
coefficients = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': grid_search.best_estimator_.coef_[0]
})

# Sort the features by absolute coefficient value to see importance
coefficients['Abs_Coefficient'] = coefficients['Coefficient'].abs()
coefficients.sort_values(by='Abs_Coefficient', ascending=False, inplace=True)

# Display the top features
print(coefficients.head(10))




