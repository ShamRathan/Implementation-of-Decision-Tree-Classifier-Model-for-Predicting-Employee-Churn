# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries .
2. Read the data frame using pandas.
3. Get the information regarding the null values present in the dataframe.
4. Apply label encoder to the non-numerical column inoreder to convert into numerical values.
5. Determine training and test data set.
6. Apply decision tree Classifier on to the dataframe.
7. Get the values of accuracy and data prediction.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: S.Sham Rathan
RegisterNumber: 212221230093  
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()

data["left"].value_counts()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

data["salary"]=le.fit_transform(data["salary"])
data.head()

x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours",
       "time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()

y=data["left"]

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy

dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
### 1. data.head()
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/aa3db765-3bf0-45c9-b538-0a2781e63cd7)

### 2. data.info()
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/f23295b3-44c3-4197-89a9-3b939a105495)

### 3. isnull() and sum()
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/368e9176-038a-403e-8ebe-906ffee88613)

### 4. data value counts()
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/e4b2d250-1139-4b3f-9297-762cc6511aa6)

### 5. data.head() for salary
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/f5f7b2ed-0374-47a0-a3bf-9cae6ae52b35)

### 6. x.head()
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/fa379e8c-1cf5-4ea6-8e70-18b7e1001512)

### 7. accuracy value
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/2f4b988a-ab28-4278-984f-017e5f7ea75e)

### 8. data prediction
![image](https://github.com/ShamRathan/Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn/assets/93587823/044f805c-1f21-449f-bf61-afb10ab083e5)




## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
