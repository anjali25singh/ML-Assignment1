# Task 4: Decision Tree Classifier - Diabetes Prediction
# Dataset: Pima Indians Diabetes Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score

# Step 1: Load the dataset
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, header=None, names=column_names)

print("First 5 rows of data:")
print(df.head())

# Step 2: Check for missing / unrealistic zero values and fix them
cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nHow many zero values in each column:")
print((df[cols_with_invalid_zeros] == 0).sum())

for col in cols_with_invalid_zeros:
    median_value = df[col].median()
    df[col] = df[col].replace(0, median_value)

# Step 3: Split into training (80%) and testing (20%) data
X = df.drop("Outcome", axis=1)   # features
y = df["Outcome"]                # target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train a Decision Tree (no depth limit)
model_full = DecisionTreeClassifier(random_state=42)
model_full.fit(X_train, y_train)
y_pred_full = model_full.predict(X_test)

print("\n--- Full Decision Tree ---")
print("Accuracy:", accuracy_score(y_test, y_pred_full))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_full))
print("Precision:", precision_score(y_test, y_pred_full))
print("Recall:", recall_score(y_test, y_pred_full))
print("F1-score:", f1_score(y_test, y_pred_full))

# Step 5: Train a Decision Tree with restricted depth (max_depth=3)
model_shallow = DecisionTreeClassifier(max_depth=3, random_state=42)
model_shallow.fit(X_train, y_train)
y_pred_shallow = model_shallow.predict(X_test)

print("\n--- Restricted Decision Tree (max_depth=3) ---")
print("Accuracy:", accuracy_score(y_test, y_pred_shallow))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_shallow))
print("Precision:", precision_score(y_test, y_pred_shallow))
print("Recall:", recall_score(y_test, y_pred_shallow))
print("F1-score:", f1_score(y_test, y_pred_shallow))

# Step 6: Compare feature importance between the two models
print("\nFeature Importance (Full Tree):")
for feature, importance in zip(X.columns, model_full.feature_importances_):
    print(f"{feature}: {importance:.4f}")

print("\nFeature Importance (max_depth=3):")
for feature, importance in zip(X.columns, model_shallow.feature_importances_):
    print(f"{feature}: {importance:.4f}")