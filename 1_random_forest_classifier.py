# task 1: Random Forest Classifier - Breast Cancer Prediction

import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Load the dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)   
y = pd.Series(data.target)                                 

print("First 5 rows of data:")
print(X.head())

# Step 2: Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Create and train the Random Forest model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 4: Make predictions on test data
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)[:, 1]   

# Step 5: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nROC-AUC Score:", roc_auc)
print("\nPrecision, Recall, F1-score:\n", classification_report(y_test, y_pred))