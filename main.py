import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
import pickle

df = pd.read_csv('loan_data_set.csv')

df['Gender'] = df['Gender'].map({'Male': 1, 'Female': 0})
df['Married'] = df['Married'].map({'Yes': 1, 'No': 0})
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})
df['Education'] = df['Education'].map({'Graduate': 1, 'Not Graduate': 0})

df = df.dropna()

df['Gender'] = df['Gender'].astype(int)
df['Married'] = df['Married'].astype(int)

X = df[['Gender', 'Married', 'ApplicantIncome', 'LoanAmount', 'Credit_History']]
y = df.Loan_Status

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = GaussianNB()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# saving the model


pickle_out = open("classifier.pkl", mode="wb")
pickle.dump(clf, pickle_out)
pickle_out.close()
