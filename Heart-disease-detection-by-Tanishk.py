import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score,precision_score,recall_score

df=pd.read_csv(r"C:\Users\tanis\Downloads\heart_cleveland_upload.csv")

print(df.head())

print(df.tail())

print(df.describe())

print(df.info())

print(df.isnull())

plt.figure(figsize=(20,20))
sns.heatmap(df.isnull(),yticklabels=False)
plt.show()

corrmat=df.corr()
features_corr=corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(df[features_corr].corr(),annot=True,cmap='RdYlGn')

df.hist(bins=10,figsize=(15,10),color='blue')
plt.show()

plt=df.plot(kind='density',subplots=True,layout=(4,4), sharex=False, sharey=False, fontsize=12, figsize=(15,10))

plt=df.plot(kind= 'box' , subplots=True, layout=(4,4), sharex=False, sharey=False,fontsize=12)

sns.set_style('whitegrid')
sns.countplot(x='target',data=df,palette='RdBu_r')

dataset = pd.get_dummies(df, columns = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal'])


standardScaler = StandardScaler()
columns_to_scale = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
dataset[columns_to_scale] = standardScaler.fit_transform(dataset[columns_to_scale])


dataset.head()


X=dataset.drop(['target'],axis=1)
y=dataset['target']


X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=.3,random_state=45)


lr=LogisticRegression()
lr.fit(X_train,y_train)

lr.score(X_test,y_test)

preducation=lr.predict(X_test)


accuracy = accuracy_score(y_test,preducation)
precision = precision_score(y_test,preducation)
recall = recall_score(y_test,preducation)
f1 = f1_score(y_test,preducation)
print("Logistic Regression Model Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


cm=confusion_matrix(y_test,preducation)
conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Paired', cbar = False,linewidths = 0.1, annot_kws = {'size':25})


import matplotlib.pyplot as plt
ax1=sns.distplot(y_test,color='r',hist=False,label='Acual value')
sns.distplot(preducation,color='b',hist=False,label='preducation',ax=ax1)
plt.title('Actual vs preducation Values')
plt.show()
plt.close()


rfc=RandomForestClassifier()
rfc.fit(X_train,y_train)


rfc.score(X_test,y_test)

preducation=lr.predict(X_test)


accuracy = accuracy_score(y_test,preducation)
precision = precision_score(y_test,preducation)
recall = recall_score(y_test,preducation)
f1 = f1_score(y_test,preducation)
print("DecisionTreeClassifier Model Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)


cm=confusion_matrix(y_test,preducation)
conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Paired', cbar = False,linewidths = 0.1, annot_kws = {'size':25})
plt.show()

ax1=sns.distplot(y_test,color='r',hist=False,label='Acual value')
sns.distplot(preducation,color='b',hist=False,label='preducation',ax=ax1)
plt.title('Actual vs preducation Values')
plt.show()
plt.close()


knn=KNeighborsClassifier(n_neighbors= 2,weights ='uniform', algorithm='auto')
knn.fit(X_train,y_train)

knn.score(X_test,y_test)

preducation=knn.predict(X_test)

accuracy = accuracy_score(y_test,preducation)
precision = precision_score(y_test,preducation)
recall = recall_score(y_test,preducation)
f1 = f1_score(y_test,preducation)
print("KNN Model Results:")
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

cm=confusion_matrix(y_test,preducation)
conf_matrix = pd.DataFrame(data = cm,columns = ['Predicted:0','Predicted:1'], index = ['Actual:0','Actual:1'])
sns.heatmap(conf_matrix, annot = True, fmt = 'd', cmap = 'Paired', cbar = False,linewidths = 0.1, annot_kws = {'size':25})
plt.show()

ax1=sns.distplot(y_test,color='r',hist=False,label='Acual value')
sns.distplot(preducation,color='b',hist=False,label='preducation',ax=ax1)
plt.title('Actual vs preducation Values')
plt.show()
plt.close()

model_names = ['LogisticRegression', 'SVM','Decision Tree', 'Random Forest','KNN']
accuracy_scores = [0.86, 0.912, 0.99,1.0,.967]
colors = [ 'paleturquoise', 'lightblue', 'skyblue','cyan','lightcyan']
plt.figure(figsize=(10, 6))
plt.bar(model_names, accuracy_scores, color=colors)
plt.xlabel('Machine Learning Models')
plt.ylabel('Accuracy Score')
plt.title('Comparison of Model Accuracy Scores')
plt.xticks(rotation=45) 
plt.tight_layout() 

plt.show()