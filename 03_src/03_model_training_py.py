# -*- coding: utf-8 -*-
"""03_model_training.py

**Model Training and Pipeline -**
- Builds and evaluates logistic regression pipeline for telecom customer churn prediction.
- Integrates preprocessing,train-test split,  training, evaluation, and business insights.
"""

# Import Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns
from google.colab import files

"""#### **1. Load Dataset-**


"""

df = pd.read_csv("telecom_churn_dataset_utf8.csv") # Upload 'telecom_churn_dataset_utf8.csv'
print("Dataset loaded, shape:", df.shape)

df.head()

df.columns = df.columns.str.strip()

df.head()

"""####**2. Data Cleaning -**"""

# Convert TotalCharges to numeric
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

# Remove missing values
df.dropna(inplace=True)

# Drop unnecessary column
df.drop(columns=['customerID'], inplace=True)

print("Cleaned Shape:", df.shape)

# Missing Value
df.isnull().sum()

"""####**3. Encode Target & Features -**"""

def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Map Churn to binary: Yes=1, No=0"""
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df

X = df.drop('Churn', axis=1)
y = df['Churn']

"""#### **4. Train-Test Split -**"""

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training shape:", X_train.shape, "Test shape:", X_test.shape)

"""#### **5. Logistic Regression Model Training -**"""

pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42))
])

# Ensure categorical features in X_train are numerically encoded
# Create a copy to perform encoding without modifying the original X_train directly
X_train_encoded = X_train.copy()

# Identify and encode object type columns
for col in X_train_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    # Fit and transform the column, ensuring it's treated as string to avoid errors with mixed types
    X_train_encoded[col] = le.fit_transform(X_train_encoded[col].astype(str))

pipeline.fit(X_train_encoded, y_train)
print(" Logistic Regression Model trained successfully")

"""#### **6. Model Evaluation -**"""

# Encode categorical features in X_test before prediction
X_test_encoded = X_test.copy()

for col in X_test_encoded.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    # Use .astype(str) to handle potential mixed types safely
    X_test_encoded[col] = le.fit_transform(X_test_encoded[col].astype(str))

y_pred = pipeline.predict(X_test_encoded)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

"""#### **7. Feature Importance / Churn Drivers -**"""

feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Coefficient': pipeline.named_steps['model'].coef_[0]
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 Features Driving Churn:")
print(feature_importance.head(10))

"""#### **8. Key Insights (Business Interpretation) -**"""

print("\n=== Key Insights (Business Interpretation) ===")

# Churn rate
churn_rate = df['Churn'].mean() * 100
print(f"- Churn rate ~{churn_rate:.1f}% → revenue risk")

# Short-tenure customers
short_tenure_churn = df[df['tenure'] < 12]['Churn'].mean() * 100
print(f"- Short-tenure customers (<12 months) more likely to churn (~{short_tenure_churn:.1f}%)")

# Top 3 features driving churn
top_features = feature_importance.head(3)['Feature'].tolist()
print(f"- Top features driving churn: {', '.join(top_features)}")

# Model accuracy
acc = accuracy_score(y_test, y_pred)
print(f"- Model accuracy: {acc:.2f} (limited due to small dataset & class imbalance)")

# Actionable insight
print("- Insights help prioritize retention strategies for high-risk customers")
