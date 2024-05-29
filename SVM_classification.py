import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# Step 1: Data Collection and Cleaning
df = pd.read_csv('CSV_folder/retractions35215.csv')

# Data Cleaning
df.dropna(subset=['CitationCount', 'Journal', 'Publisher', 'Author', 'ArticleType', 'Country'], inplace=True)

# Feature selection
X = df[['Journal', 'Publisher', 'Author', 'ArticleType', 'Country']]
y = df['CitationCount']

# Convert categorical variables to numeric using LabelEncoder
label_encoders = {}
for column in X.columns:
    le = LabelEncoder()
    X.loc[:, column] = le.fit_transform(X[column])
    label_encoders[column] = le

# Bin the target variable into 3 categories (low, medium, high)
unique_bins = pd.qcut(y, q=3, retbins=True, duplicates='drop')[1]
n_bins = len(unique_bins) - 1
labels = ["low", "medium", "high"][:n_bins]

y_binned = pd.qcut(y, q=n_bins, labels=labels, duplicates='drop')

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binned, test_size=0.2, random_state=42)

# Step 2: Train the SVM classifier
svm_classifier = SVC(kernel='linear')
svm_classifier.fit(X_train, y_train)

# Step 3: Make predictions
y_pred = svm_classifier.predict(X_test)

# Step 4: Evaluate the classifier
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_report(y_test, y_pred))
