# -*- coding: utf-8 -*-


# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report


df = pd.read_csv(r'C:\Users\ASUS\OneDrive\Desktop\Credit retention and reward optimization using Ml\TEST\modified_dataset_with_purchase_locations.csv')
df.head()

# Step 3: Prepare the data for churn prediction (Assume 'CHURN' is the target column)
X = df.drop(columns=['CHURN', 'CUST_ID','CREDIT_LIMIT','MINIMUM_PAYMENTS','PAYMENTS', 'PURCHASE_LOCATION'],axis=1)  # Drop target and non-relevant columns
y = df['CHURN']  # Target variable: 1 = Churn, 0 = Not Churn
X.shape

# Step 4: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Step 5: Train the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100)
rf_classifier.fit(X_train, y_train)

# Step 6: Predict churn on the test set
y_pred = rf_classifier.predict(X_test)

# Step 7: Evaluate the churn prediction model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy: {:.2f}%".format(accuracy * 100))
print("Precision: {:.2f}%".format(precision * 100))
print("Recall: {:.2f}%".format(recall * 100))
print("F1 Score: {:.2f}%".format(f1 * 100))

# Confusion matrix for visualization
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Churn', 'Churn'], yticklabels=['Not Churn', 'Churn'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

X.shape

rf_classifier.predict([[40.900749,0.818182,95.4,0,95.4,0,0.166667,0,0.083333,0,0,2,0,12]])

