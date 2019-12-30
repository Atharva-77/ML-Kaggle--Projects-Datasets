import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

df=pd.read_csv('train.csv')
df.head(5)

x=df.iloc[:,[0,2,3,4,5,6,7,8,9,11]]
x.head(5)

y=df.iloc[:,1]
y.head(5)

x.isnull().sum(axis=0)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()

x.iloc[:, 3] = labelencoder.fit_transform(x.iloc[:, 3])

x['Embarked'] = labelencoder.fit_transform(df['Embarked'].astype(str))
x.iloc[:, 9] = labelencoder.fit_transform(x.iloc[:, 9])
x.iloc[:, 7] = labelencoder.fit_transform(x.iloc[:, 7])#ticket is encoded


x.drop(['Name'], axis = 1, inplace = True) 


from sklearn.preprocessing import Imputer
imputer=Imputer(strategy='mean')
imputer=imputer.fit(x.iloc[:,[3]])
x.iloc[:,[3]]=imputer.transform(x.iloc[:,[3]])#age

imputer1=Imputer(strategy='most_frequent')
imputer1=imputer1.fit(x.iloc[:,[8]])
x.iloc[:,[8]]=imputer1.transform(x.iloc[:,[8]])# embarked

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)

import statsmodels.api as sm
x=np.append(arr=np.ones((891,1)),values=x,axis=1)


xopt=x[:,[0,1,2,3,4,5,6,7,8,9]]
regosl=sm.OLS(endog=y,exog=xopt).fit()
regosl.summary()

xopt=x[:,[0,2,3,4,5,6,7,8,9]]
regosl=sm.OLS(endog=y,exog=xopt).fit()
regosl.summary()

xopt=x[:,[0,2,3,4,5,7,8,9]]
regosl=sm.OLS(endog=y,exog=xopt).fit()
regosl.summary()

xopt=x[:,[0,2,3,4,5,7,9]]
regosl=sm.OLS(endog=y,exog=xopt).fit()
regosl.summary()


xopt=x[:,[0,2,3,4,5,9]]
regosl=sm.OLS(endog=y,exog=xopt).fit()
regosl.summary()

X_train=pd.DataFrame(X_train).to_numpy()
X_test=pd.DataFrame(X_test).to_numpy()
X_trainNew=X_train[:,[0,2,3,4,5,8]]
X_testNew=X_test[:,[0,2,3,4,5,8]]

X_trainNew = X_trainNew.astype('float64') 
X_testNew = X_testNew.astype('float64') 

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_trainNew, y_train)
ypred1=regressor.predict(X_testNew)

k=0
for i in ypred1:
    if i>=0.5:
        ypred1[k]=1
    else:
        ypred1[k]=0
    k=k+1

ypred1=pd.Series(ypred1)
ypred1 = ypred1.astype('Int64')

pred_res1=0
pred_res0=0
for i in ypred1:
    if i>=0.5:
        pred_res1=pred_res1+1
    else:
        pred_res0=pred_res0+1

print(pred_res1,pred_res0)

acc_res1=0
acc_res0=0
for i in y_test:
    if i>=0.5:
        acc_res1=acc_res1+1
    else:
        acc_res0=acc_res0+1

print(acc_res1,acc_res0)


dftest=pd.read_csv('test.csv')
dftest.head(5)
x1=dftest.iloc[:,[0,1,2,3,4,5,6,7,8,9,10]]

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
x1.iloc[:, 3] = labelencoder.fit_transform(x1.iloc[:, 3])
x1.iloc[:, 10] = labelencoder.fit_transform(x1.iloc[:, 10])#embarked is encoded


x1.drop(['Name'], axis = 1, inplace = True) 
x1.drop(['Ticket'], axis = 1, inplace = True) 
x1.iloc[:,[3]]=imputer.transform(x1.iloc[:,[3]])


x1=pd.DataFrame(x1).to_numpy()
X_testNew1=x1[:,[0,2,3,4,5,8]]

X_testNew1=pd.DataFrame(X_testNew1)
X_testNew1.isnull().sum(axis=0)

ypred2=regressor.predict(X_testNew1)
k=0
for i in ypred2:
    if i>=0.5:
        ypred2[k]=1
    else:
        ypred2[k]=0
    k=k+1
    
ypred2 = ypred2.astype(int) 
ypred1 = ypred1.astype(int)


from sklearn.metrics import accuracy_score 
accuracy_score(y_test,ypred1) 

PassengerId=dftest['PassengerId']
PassengerId
submission=pd.DataFrame({'PassengerId':PassengerId})
submission['Survived']=ypred2
submission.to_csv('TITanic2.csv',index=False)
#Accuracy is 0.7703
