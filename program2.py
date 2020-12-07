import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

data=pd.read_csv('train_lyst4393 (2).csv')
#print(data)
del data['Cabin']
sns.heatmap(data.isnull())
#plt.show()

def impute_age(cols):
    Age=cols[0]
    Pclass=cols[1]
    if pd.isnull(Age):
        if Pclass==1:
            return 37
        elif Pclass==2:
            return 29
        else:
            return 24
    else:
        return Age

data['Age']=data[['Age','Pclass']].apply(impute_age,axis=1)
sns.heatmap(data.isnull())
#plt.show()
data.dropna(inplace=True)
sns.heatmap(data.isnull())
#plt.show()
#print(data.columns)
del data['Name']
#print(data.columns)

sex=pd.get_dummies(data['Sex'],drop_first=True)
#print(sex)
embark=pd.get_dummies(data['Embarked'],drop_first=True)
#print(embark)
del data['Sex']
del data['Embarked']
del data['Ticket']

data=pd.concat([data,sex,embark],axis=1)

#print(data)
feature_cols=data[['PassengerId',  'Pclass', 'Age', 'SibSp', 'Parch', 'Fare','male', 'Q', 'S']]
x=feature_cols
#print(x)
y=data['Survived']
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
#print(y_pred)
val=pd.DataFrame(y_pred,y_test)
#print(val)

from sklearn import metrics
cnf_matrix=metrics.confusion_matrix(y_test,y_pred)
print(cnf_matrix)
sns.heatmap(pd.DataFrame(cnf_matrix),annot=True,fmt='g')
#plt.show()

print(metrics.accuracy_score(y_test,y_pred))
print(metrics.precision_score(y_test,y_pred))
print(metrics.recall_score(y_test,y_pred))

 
    
