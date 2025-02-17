#!/usr/bin/env python
# coding: utf-8

# In[66]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[67]:


df = pd.read_csv(r"C:\Users\emilb\OneDrive\Desktop\iris (1).csv")


# In[68]:


df


# In[69]:


df.head()


# In[70]:


df.shape


# In[71]:


df.columns


# In[72]:


df.dtypes


# In[73]:


df.info()


# In[74]:


df.describe()


# # Data Preprocessing

# In[75]:


df.isna().sum()


# In[76]:


num=df.select_dtypes(include="float64")


# In[77]:


num


# In[78]:


num.hist(figsize=[20,16])
plt.show()


# In[79]:


df["SL"]=df["SL"].fillna(df["SL"].mean())


# In[80]:


df["SW"]=df["SW"].fillna(df["SL"].mean())


# In[81]:


df["PL"]=df["PL"].fillna(df["SL"].median())


# In[82]:


df.isna().sum()


# In[83]:


df.duplicated().sum()


# In[84]:


df = df.drop_duplicates()


# In[85]:


sns.boxplot(df)
plt.show()


# In[86]:


Q1=df["SW"].quantile(0.25)
Q3=df["SW"].quantile(0.75)
IQR=Q3-Q1
lower_bound=Q1-1.5*IQR
upper_bound=Q3+1.5*IQR

#find outlires 
outliers=df[(df["SW"]<lower_bound)|(df["SW"]>upper_bound)]
outliers


# In[ ]:





# In[87]:


df=df[(df["SW"]>=lower_bound)&(df["SW"]<=upper_bound)]
df.shape


# In[88]:


sns.boxplot(df)
plt.show()


# In[89]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["Classification"]=le.fit_transform(df["Classification"])


# In[90]:


df.head()


# In[91]:


type(df)


# In[92]:


df=pd.DataFrame(df)
df


# In[93]:


df.columns = ['SL', 'SW', 'PL', 'PW', 'Classification']


# In[94]:


df


# In[95]:


corr = df.corr()

# plot the heatmap
sns.heatmap(corr,annot=True,cmap="YlGn")


# In[96]:


y=df["Classification"]
x=df.drop("Classification",axis=1)


# In[97]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,confusion_matrix
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=.25)


# # Logistic Regression

# In[98]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(x_train,y_train)
y_log_pred =lr.predict(x_test)
y_log_pred


# In[99]:


acc_log = accuracy_score(y_test,y_log_pred)
pre_log = precision_score(y_test,y_log_pred,average='weighted')
re_log = recall_score(y_test,y_log_pred,average='weighted')
f1_log = f1_score(y_test,y_log_pred,average='weighted')


# In[100]:


print('Accuracy: ',acc_log)
print('Precision: ',pre_log)
print('Recall: ',re_log)
print('F1: ',f1_log)
     


# # KNN

# In[101]:


from sklearn.neighbors import KNeighborsClassifier
metric_k=[]
neighbors =np.arange(3,15)
for k in neighbors:
    k_model=KNeighborsClassifier(n_neighbors=k,metric="euclidean")
    k_model.fit(x_train,y_train)
    y_pred_knn=k_model.predict(x_test)
    acc_knn=accuracy_score(y_test,y_pred_knn)
    metric_k.append(acc_knn)
plt.plot(neighbors,metric_k,'o-')
plt.xlabel('k value')
plt.ylabel('accuracy')


# In[102]:


knn=KNeighborsClassifier(n_neighbors=5,metric="minkowski")
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)
acc_knn = accuracy_score(y_test,y_pred_knn)
pre_knn = precision_score(y_test,y_pred_knn, average='weighted')
re_knn = recall_score(y_test,y_pred_knn, average='weighted')
f1_knn = f1_score(y_test,y_pred_knn, average='weighted')


# In[103]:


print('Accuracy: ',acc_knn)
print('Precision: ',pre_knn)
print('Recall: ',re_knn)
print('F1: ',f1_knn)


# # Decision Tree model

# In[104]:


from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)
acc_dt = accuracy_score(y_test,y_pred_dt)
pre_dt = precision_score(y_test,y_pred_dt, average='weighted')
re_dt = recall_score(y_test,y_pred_dt, average='weighted')
f1_dt = f1_score(y_test,y_pred_dt, average='weighted')
     


# In[105]:


print('Accuracy: ',acc_dt)
print('Precision: ',pre_dt)
print('Recall: ',re_dt)
print('F1: ',f1_dt)
     


# # Randanom Forest Model

# In[106]:


from sklearn.ensemble import RandomForestClassifier
rf= RandomForestClassifier()
rf.fit(x_train,y_train)
y_pred_rf = rf.predict(x_test)   
acc_rf = accuracy_score(y_test,y_pred_rf)
pre_rf = precision_score(y_test,y_pred_rf, average='weighted')
re_rf = recall_score(y_test,y_pred_rf, average='weighted')
f1_rf = f1_score(y_test,y_pred_rf, average='weighted')
     


# In[107]:


print('Accuracy: ',acc_rf)
print('Precision: ',pre_rf)
print('Recall: ',re_rf)
print('F1: ',f1_rf)


# # SVM

# In[108]:


from sklearn.svm import SVC
sv= SVC(kernel='rbf')
sv.fit(x_train,y_train)
y_pred_sv = sv.predict(x_test)
acc_sv = accuracy_score(y_test,y_pred_sv)
pre_sv = precision_score(y_test,y_pred_sv, average='weighted')
re_sv = recall_score(y_test,y_pred_sv, average='weighted')
f1_sv = f1_score(y_test,y_pred_sv, average='weighted')
     


# In[109]:


print('Accuracy: ',acc_sv)
print('Precision: ',pre_sv)
print('Recall: ',re_sv)
print('F1: ',f1_sv)


# # Naive BAyers Model

# In[113]:


from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)
y_pred_nb=gnb.predict(x_test)
acc_nb = accuracy_score(y_test,y_pred_nb)
pre_nb = precision_score(y_test,y_pred_nb, average='weighted')
re_nb= recall_score(y_test,y_pred_nb, average='weighted')
f1_nb = f1_score(y_test,y_pred_nb, average='weighted')
     


# In[114]:


print('Accuracy: ',acc_nb)
print('Precision: ',pre_nb)
print('Recall: ',re_nb)
print('F1: ',f1_nb)


# In[115]:


Accuracy = pd.DataFrame({'Models': ['Logistic Regression', 'KNN','Decision Tree', 'Random Forest','SVM_Linear', "Naive BAyers Model"],
                         'Accuracies':[acc_log,acc_knn,acc_dt,acc_rf,acc_sv,acc_nb]})


# In[116]:


Accuracy


# from this 6 model KNN achieved 100% accuracy,Logistic Regression, Decision Tree, and Random Forest have similar accuracies (97.14%),SVM (Linear) has the lowest accuracy (94.29%)

# In[ ]:





# In[ ]:




