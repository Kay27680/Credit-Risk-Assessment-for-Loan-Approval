import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

#Load and Preprocess Data
# Load the dataset
data = pd.read_csv('loan_data.csv')

# Display basic info
print(data.info())

# Feature engineering: Handle missing values
data.fillna(data.median(), inplace=True)

# Encode categorical variables
data = pd.get_dummies(data, drop_first=True)

# Separate features (X) and target (y)
X = data.drop('default_status', axis=1)  # Assuming 'default_status' is the target
y = data['default_status']

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Standardize Numerical Features
# Standardize the features for better performance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#Train a Machine Learning Model
# Initialize and train a Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]  # For ROC-AUC

# Evaluate the Model
# Classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_pred_proba)
print(f"ROC-AUC Score: {roc_auc:.2f}")

#Feature Importance
# Feature importance for interpretability
importance = model.feature_importances_
feature_names = X.columns

# Print sorted feature importance
sorted_importance = sorted(zip(feature_names, importance), key=lambda x: x[1], reverse=True)
print("Feature Importance:")
for feature, imp in sorted_importance:
    print(f"{feature}: {imp:.4f}")
