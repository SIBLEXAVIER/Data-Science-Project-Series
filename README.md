# Data-Science-Project-Series
CODE OF BREAST CANCER PREDICTION
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

df=pd.read_csv("data.csv")

df

column_names = df.columns
print(column_names)

df.head()

df.info()

df.describe()

df.shape

missing_values = df.isnull().sum()
print(missing_values)


df.isnull().sum()

df.isna().sum()

df['diagnosis'].value_counts()

feature extraction

X = df.drop(['id', 'diagnosis'], axis=1)  # Drop 'id' and 'diagnosis' columns
y = df['diagnosis']

df= pd.DataFrame(df['diagnosis'])

encoded_data = pd.get_dummies(df,'daignosis')

print("Original Data:")
print(df)
print("\nEncoded Data:")
print(encoded_data)

encoded_data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

using SVM

from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

svm_model = SVC(kernel='linear')

svm_model.fit(X_train, y_train)

y_pred = svm_model.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Calculate the  svm accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy of SVM model: {accuracy:.2f}")
