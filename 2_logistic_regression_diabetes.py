# Task 2: Logistic Regression - Diabetes Prediction
# Dataset: Pima Indians Diabetes Dataset

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Load the dataset
column_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
                 "Insulin", "BMI", "DiabetesPedigreeFunction", "Age", "Outcome"]

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
df = pd.read_csv(url, header=None, names=column_names)

print("First 5 rows of data:")
print(df.head())

# Step 2: Check for missing / zero values
# In this dataset, a 0 in these columns is not realistic -> treat as missing
cols_with_invalid_zeros = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]
print("\nHow many zero values in each column:")
print((df[cols_with_invalid_zeros] == 0).sum())

# Replace 0 with the median value of that column
for col in cols_with_invalid_zeros:
    median_value = df[col].median()
    df[col] = df[col].replace(0, median_value)

# Step 3: Split features (X) and target (y)
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Step 4: Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 5: Scale the features (important for Logistic Regression)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 6: Train the Logistic Regression model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# Step 7: Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Step 8: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nROC-AUC Score:", roc_auc)
print("\nPrecision, Recall, F1-score:\n", classification_report(y_test, y_pred))

# Step 9: Look at the model coefficients
print("\nModel Coefficients (which features matter most):")
for feature, coef in zip(X.columns, model.coef_[0]):
    print(f"{feature}: {coef:.4f}")

print("\nA positive number means that feature increases diabetes risk.")
print("A negative number means it decreases diabetes risk.")