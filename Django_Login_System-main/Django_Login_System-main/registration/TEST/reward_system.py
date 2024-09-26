# -*- coding: utf-8 -*-




# Step 1: Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv(r'Django_Login_System-main/registration/TEST/modified_dataset_with_purchase_locations.csv')
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

# Step 8: Define the reward suggestion function based on churn prediction and purchase location
def suggest_rewards(row):
    reward = None
    if row['CHURN'] == 1:  # Customer is predicted to churn
        if row['PURCHASE_LOCATION'] == 'Gas Station':
            reward = 'Free gas on next purchase'
        elif row['PURCHASE_LOCATION'] == 'Online Shopping':
            reward = 'Discount on next online purchase'
        elif row['PURCHASE_LOCATION'] == 'Grocery Store':
            reward = 'Discount on next grocery purchase'
        elif row['PURCHASE_LOCATION'] == 'Restaurants':
            reward = 'Free meal or discount at restaurant'
        elif row['PURCHASE_LOCATION'] == 'Electronics Store':
            reward = 'Discount on next electronics purchase'
        elif row['PURCHASE_LOCATION'] == 'Pharmacy':
            reward = 'Free or discounted pharmacy items'
        else:
            reward = 'Special discount'
    else:  # Customer is not predicted to churn
        reward = 'Cashback on next purchase'

    # Print the customer ID and the reward for debugging purposes
    print(f"Customer ID: {row['CUST_ID']}, Reward: {reward}")

    return reward

# Step 9: Combine churn prediction with reward suggestion
# Note: We still need PURCHASE_LOCATION for the reward system, so it's not dropped from df
df['CHURN_PREDICTED'] = rf_classifier.predict(df.drop(columns=['CHURN', 'CUST_ID','CREDIT_LIMIT','MINIMUM_PAYMENTS','PAYMENTS', 'PURCHASE_LOCATION']))

X.shape

# Apply the reward suggestion function (PURCHASE_LOCATION is still needed for rewards)
df['REWARD_SUGGESTION'] = df.apply(suggest_rewards, axis=1)

