from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm

import joblib

df = pd.read_csv("loan.csv")

df.head(5)

df.info()

df.isnull().sum()

df['loanamount_log']=np.log(df['LoanAmount'])
df['loanamount_log'].hist(bins=20)

df.isnull().sum()

df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
df['TotalIncome_log']=np.log(df['TotalIncome'])
df['TotalIncome_log'].hist(bins=20)

df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
df['Married'].fillna(df['Married'].mode()[0],inplace=True)
df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
df.loanamount_log = df.loanamount_log.fillna(df.loanamount_log.mean())

df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0],inplace=True)
df['Credit_History'].fillna(df['Credit_History'].mode()[0],inplace=True)

df.isnull().sum()

x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
y = df.iloc[:,12].values

x

y

print("percent of missing gender is %2f%%" %((df['Gender'].isnull().sum()/df.shape[0])*100))

print("number of people who take loan as group by gender :- ")
print(df['Gender'].value_counts())
sns.countplot(x='Gender',data=df,palette = 'Set1')

print("number of people who take loan as group by marital status :- ")
print(df['Married'].value_counts())
sns.countplot(x='Married',data=df,palette = 'Set1')

print("number of people who take loan as group by dependents :- ")
print(df['Dependents'].value_counts())
sns.countplot(x='Dependents',data=df,palette = 'Set1')

print("number of people who take loan as group by self employed :- ")
print(df['Self_Employed'].value_counts())
sns.countplot(x='Self_Employed',data=df,palette = 'Set1')

print("number of people who take loan as group by Loanamount :- ")
print(df['LoanAmount'].value_counts())
sns.countplot(x='LoanAmount',data=df,palette = 'Set1')

print("number of people who take loan as group by Credit history :- ")
print(df['Credit_History'].value_counts())
sns.countplot(x='Credit_History',data=df,palette = 'Set1')

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=1)

LabelEncoder_X = LabelEncoder()

for i in range(0,5):
    X_train[:,i]= LabelEncoder_X.fit_transform(X_train[:,i])
    X_train[:,7]= LabelEncoder_X.fit_transform(X_train[:,7])
    
X_train

LabelEncoder_y = LabelEncoder()
y_train = LabelEncoder_y.fit_transform(y_train)
    
y_train

for i in range(0,5):
    X_test[:,i]=LabelEncoder_X.fit_transform(X_test[:,i])
    X_test[:,7]=LabelEncoder_X.fit_transform(X_test[:,7])

X_test

LabelEncoder_y = LabelEncoder()
y_test = LabelEncoder_y.fit_transform(y_test)

y_test



ss = StandardScaler()

X_train = ss.fit_transform(X_train)
X_test = ss.fit_transform(X_test)



rf_clf = RandomForestClassifier()
rf_clf.fit(X_train, y_train)

from sklearn import metrics
y_pred = rf_clf.predict(X_test)

print("According to Random Forest Classifier is ", metrics.accuracy_score(y_pred, y_test))

y_pred

nb_clf = GaussianNB()
nb_clf.fit(X_train, y_train)

y_pred = nb_clf.predict(X_test)
print("According to GaussianNB is ", metrics.accuracy_score(y_pred, y_test))

from sklearn.tree import DecisionTreeClassifier
dt_clf = DecisionTreeClassifier()
dt_clf.fit(X_train, y_train)

y_pred = dt_clf.predict(X_test)
print("According to Decision Tree is ", metrics.accuracy_score(y_pred, y_test))

joblib.dump(rf_clf, 'rf_model.sav')
joblib.dump(nb_clf, 'nb_model.sav')
joblib.dump(dt_clf, 'dt_model.sav')