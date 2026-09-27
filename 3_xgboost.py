# Task 3: XGBoost Classifier

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

# Step 1: Load the dataset
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)

print("First 5 rows of data:")
print(df.head())

# Step 2: Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Fill missing Age with the median age
df["Age"] = df["Age"].fillna(df["Age"].median())

# Fill missing Embarked with the most common value
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Drop columns that are not useful for predicting survival
df = df.drop(columns=["Cabin", "Ticket", "Name", "PassengerId"])

# Step 3: Convert text columns into numbers
# Sex: male/female -> 1/0
df["Sex"] = df["Sex"].map({"male": 1, "female": 0})

# Embarked: turn into separate 0/1 columns
df = pd.get_dummies(df, columns=["Embarked"], drop_first=True)

# Step 4: Split features (X) and target (y)
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Step 5: Split into training and testing data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Step 7: Train the XGBoost model
model = XGBClassifier(n_estimators=100, max_depth=4, random_state=42, eval_metric="logloss")
model.fit(X_train_scaled, y_train)

# Step 8: Make predictions
y_pred = model.predict(X_test_scaled)
y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

# Step 9: Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print("\nAccuracy:", accuracy)
print("\nConfusion Matrix:\n", cm)
print("\nROC-AUC Score:", roc_auc)
print("\nPrecision, Recall, F1-score:\n", classification_report(y_test, y_pred))

# Step 10: Feature importance (which columns mattered most)
print("\nFeature Importance:")
for feature, importance in zip(X.columns, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")