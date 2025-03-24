from django.shortcuts import render

# Create your views here.

def index(request):
    return render(request, 'index.html')

def calculator(request):
    return render(request, 'calculator.html')

def result(request):
    if request.method == "POST":
        import numpy as np
        import pandas as pd
        from sklearn import svm
        import joblib

        from sklearn.preprocessing import LabelEncoder
        from sklearn.preprocessing import StandardScaler


        lis = [
            request.POST.get('id', 0),
            request.POST.get('gender', 'Male'),
            request.POST.get('mstatus', 'No'),
            request.POST.get('dependents', '0'),
            request.POST.get('education', 'Not Graduate'),
            request.POST.get('profession', 'No'),
            int(request.POST.get('income', 0)),
            float(request.POST.get('cincome', 0)),
            float(request.POST.get('amount', 0)),
            float(request.POST.get('term', 0)),
            float(request.POST.get('chistory', 0)),
            request.POST.get('parea', 'Rural'),
            request.POST.get('loan', 'N'),
        ]

        df = pd.DataFrame([lis], columns=['Loan_ID', 'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 'Loan_Amount_Term', 'Credit_History', 'Property_Area', 'Loan_Status'])

        df['loanamount_log']=np.log(df['LoanAmount'])

        df['TotalIncome']=df['ApplicantIncome']+df['CoapplicantIncome']
        df['TotalIncome_log']=np.log(df['TotalIncome'])

        df['Gender'].fillna(df['Gender'].mode()[0],inplace=True)
        df['Married'].fillna(df['Married'].mode()[0],inplace=True)
        df['Self_Employed'].fillna(df['Self_Employed'].mode()[0],inplace=True)
        df['Dependents'].fillna(df['Dependents'].mode()[0],inplace=True)

        df.LoanAmount = df.LoanAmount.fillna(df.LoanAmount.mean())
        df.loanamount_log = df.loanamount_log.fillna(df.loanamount_log.mean())

        x = df.iloc[:,np.r_[1:5,9:11,13:15]].values
        y = df.iloc[:,12].values
        
        LabelEncoder_X = LabelEncoder()

        for i in range(0,5):
            x[:,i]= LabelEncoder_X.fit_transform(x[:,i])
            x[:,7]= LabelEncoder_X.fit_transform(x[:,7])

        LabelEncoder_y = LabelEncoder()
        y = LabelEncoder_y.fit_transform(y)

        ss = StandardScaler()

        x = ss.fit_transform(x)

        rf_mod = joblib.load('rf_model.sav')
        nb_mod = joblib.load('nb_model.sav')
        dt_mod = joblib.load('dt_model.sav')

        rfa = rf_mod.predict(x)
        nba = nb_mod.predict(x)
        dta = dt_mod.predict(x)

        return render(request, 'result.html', {'rfa': rfa, 'nba': nba, 'dta': dta})

    return render(request, 'result.html')