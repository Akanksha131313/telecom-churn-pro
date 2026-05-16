# -*- coding: utf-8 -*-
"""02_model_evaluation.ipynb

**Model Evaluation Utility-**  
- Provides standardized evaluation metrics for churn prediction models.
- Outputs Accuracy, Classification Report, and Confusion Matrix for recruiter-friendly assessment.
"""

# Import Libraries
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

from google.colab import files
uploaded = files.upload()

df = pd.read_csv("telecom_churn_dataset_utf8.csv")

# Function: Evaluate Model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates a trained model and prints key metrics.

    Args:
        model : Trained sklearn model or pipeline
        X_test : Features for testing
        y_test : True labels for testing
    """

# 1️. Generate predictions first
predictions = model.predict(X_test)

# 2️. Then calculate accuracy
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# 3. Optional: Classification report
from sklearn.metrics import classification_report, confusion_matrix
print("\nClassification Report:\n", classification_report(y_test, predictions))

# 4. Optional: Confusion Matrix
cm = confusion_matrix(y_test, predictions)
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.show()
