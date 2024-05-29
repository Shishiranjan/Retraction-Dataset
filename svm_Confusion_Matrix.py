import pandas as pd 
from sklearn import svm, metrics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC 

from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

import matplotlib.pyplot as plt 


# A. Load dataset
df = pd.read_csv('retractions.csv', lineterminator='\n') 


# B. Prepare dataset for training
# splitting data into 70% training and 30% test


df.dropna(subset=['CitationCount', 'Journal', 'Publisher', 'Author', 'ArticleType', 'Country'], inplace=True) 

X = df[['Journal', 'Publisher', 'Author', 'ArticleType', 'Country']] 

y = df['CitationCount']



unique_bins = pd.qcut(y, q=3, retbins=True, duplicates='drop')[1] 

n_bins = len(unique_bins) - 1 

labels = ["low", "medium", "high"][:n_bins] 
y_binned = pd.qcut(y, q=n_bins, labels=labels, duplicates='drop') 


scaler = StandardScaler() 

X_scaled = scaler.fit_transform(X) 

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_binned, test_size=0.2, random_state=42)

# C. Build an SVM model
# create an instance of the SVM classifier with a Linear kernel
model = svm.SVC(kernel='linear')

# train the model using the training set
model.fit(X_train, y_train)

# predict the classes in the test set
y_pred = model.predict(X_test)


# D. Evaluate the model
# accuracy
accuracy = accuracy_score(y_test, y_pred) 

print("Accuracy:", accuracy) 

print("Accuracy: %.3f%%" % (metrics.accuracy_score(y_test, y_pred)*100))

# precision
print("Precision: %.3f " % metrics.precision_score(y_test, y_pred, pos_label=0))

# recall
print("Recall: %.3f" % metrics.recall_score(y_test, y_pred, pos_label=0))

# F1 (F-Measure)
print("F1: %.3f" % metrics.f1_score(y_test, y_pred, pos_label=0))

print("\nClassification Report:\n", classification_report(y_test, y_pred)) 



# compute Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)

# pretty print Confusion Matrix as a heatmap
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=y_test)
disp.plot()
plt.show()
