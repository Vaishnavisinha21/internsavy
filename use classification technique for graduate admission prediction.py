#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


data = pd.read_csv("C:/Users/vaish/Downloads/Admission_Predict.csv")


# # 1.display top 5 rows of the dataset

# In[3]:


data.head()


# # 2. check last 5 rows of the datset

# In[4]:


data.tail()


# # 3. find shape of our dataset(Number of rows and number of columns)

# In[5]:


data.shape


# # 4. get info about our dataset like total numbet of rows, total number of columns, datatypes of each column and memory requirement

# In[6]:


data.info()


# # 5. check null values in the dataset

# In[7]:


data.isnull().sum()


# # 6. get overall statistics about the dataset

# In[8]:


data.describe()


# # 7. Dropping irrevelant features

# In[9]:


data.columns


# In[10]:


data= data.drop('Serial No.', axis=1)


# In[11]:


data.columns


# # 8. store feature matrix in X and response(target) in vector Y 

# In[12]:


data.head(1)


# In[13]:


data.columns


# In[14]:


X= data.drop('Chance of Admit ', axis=1)


# In[15]:


X


# In[16]:


y= data.drop('Chance of Admit ',axis=1)


# In[17]:


y= data['Chance of Admit ']


# In[18]:


y


# # 9. splitting the dataset imto the training set and test set

# In[19]:


from sklearn.model_selection import train_test_split


# In[20]:


X_train, X_test, y_train, y_test= train_test_split(X,y,test_size=0.20,random_state=42)


# In[21]:


y_train


# # 10. feature scaling

# In[22]:


data.head()


# In[23]:


from sklearn.preprocessing import StandardScaler


# In[24]:


sc= StandardScaler()


# In[25]:


X_train=sc.fit_transform(X_train)
X_test=sc.transform(X_test)


# In[26]:


X_train


# # 11. import the models

# In[27]:


data.head()


# In[28]:


from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor


# # 12. model training

# In[29]:


lr= LinearRegression()
lr.fit(X_train,y_train)

svm=SVR()
svm.fit(X_train,y_train)

rf= RandomForestRegressor()
rf.fit(X_train,y_train)

gr=GradientBoostingRegressor()
gr.fit(X_train,y_train)


# # 13. predicting on test data

# In[30]:


y_pred1= lr.predict(X_test)
y_pred2= svm.predict(X_test)
y_pred3= rf.predict(X_test)
y_pred4= gr.predict(X_test)


# # 14. Evaluating the algorithm

# In[31]:


from sklearn import metrics


# In[32]:


score1= metrics.r2_score(y_test,y_pred1)
score2= metrics.r2_score(y_test,y_pred2)
score3= metrics.r2_score(y_test,y_pred3)
score4= metrics.r2_score(y_test,y_pred4)


# In[33]:


print(score1,score2,score3,score4)


# In[34]:


final_data = pd.DataFrame({'Models':['LR','SVR','RF','GR'],'R2_SCORE':[score1,score2,score3,score4]})


# In[35]:


final_data


# # Classification

# In[36]:


data.head()


# In[37]:


import numpy as np


# In[38]:


y_train = [1 if value>0.8 else 0 for value in y_train]
y_test = [1 if value>0.8 else 0 for value in y_test]

y_train = np.array(y_train)
y_test = np.array(y_test)


# In[39]:


y_train


# # 15. import the models

# In[40]:


from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score


# # 16. model training and evaluation

# In[41]:


lr = LogisticRegression()
lr.fit(X_train,y_train)
y_pred1= lr.predict(X_test)
print(accuracy_score(y_test,y_pred1))


# In[42]:


svm=svm.SVC() 
svm.fit(X_train,y_train) 
y_pred2= svm.predict(X_test) 
print(accuracy_score(y_test,y_pred2))


# svm=svm.SVC()
# svm.fit(X_train,y_train)
# y_pred2= svm.predict(X_test)
# print(accuracy_score(y_test,y_pred2))

# In[43]:


knn = KNeighborsClassifier()
knn.fit(X_train,y_train)
y_pred3= knn.predict(X_test)
print(accuracy_score(y_test,y_pred3))


# In[44]:


rf= RandomForestClassifier()
rf.fit(X_train,y_train)
y_pred4=rf.predict(X_test)
print(accuracy_score(y_test,y_pred4))
    
    


# In[45]:


gr= GradientBoostingClassifier()
gr.fit(X_train,y_train)
y_pred5= gr.predict(X_test)
print(accuracy_score(y_test,y_pred5))


# In[48]:


final_data= pd.DataFrame({'Models':['LR','SVC','KNN','RF','GBC'],
                          'ACC_SCORE':[accuracy_score(y_test,y_pred1),
                                       accuracy_score(y_test,y_pred2),
                                       accuracy_score(y_test,y_pred3),
                                       accuracy_score(y_test,y_pred4),
                                       accuracy_score(y_test,y_pred5)]})


# In[49]:


final_data


# # 17.Save the model

# In[55]:


data.columns


# In[58]:


X= data.drop('Chance of Admit ',axis=1)


# In[59]:


X


# In[60]:


y= data['Chance of Admit ']


# In[62]:


y= [1 if value>0.8 else 0 for value in y]


# In[63]:


y


# In[64]:


y=np.array(y)


# In[65]:


y


# In[66]:


X= sc.fit_transform(X)


# In[67]:


X


# In[68]:


gr= GradientBoostingClassifier()
gr.fit(X,y)


# In[69]:


import joblib


# In[70]:


joblib.dump(gr,'admission_model')


# In[72]:


model = joblib.load('admission_model')


# In[73]:


data.columns


# In[75]:


model.predict(sc.transform([[337,118,4,4.5,4.5,9.65,1]]))


# # GUI

# In[ ]:


from tkinter import *
import joblib
from sklearn.preprocessing import StandardScaler


# In[ ]:


def show_entry():
    
    p1=float(e1.get())
    p2=float(e2.get())
    p3=float(e3.get())
    p4=float(e4.get())
    p5=float(e5.get())
    p6=float(e6.get())
    p7=float(e7.get())
    
    model=joblib.load('admission_model')
    result= model.predict(sc.transform([p1,p2,p3,p4,p5,p6,p7]))
    
    if result == 1:
        label(master,text="High Chance of getting admission").grid(row=31)
    else:
        label(master,text="you may get admission").grid(row=31)
        
master= Tk()  
master.title("Graduate Admission Analysis and Prediction")
label= Label(master,text="Graduate Admission Analysis and Prediction",
             fg="white").grid(row=0,columnspan=2)

Label(master,text= "Enter your GRE score").grid(row=1)
Label(master,text= "Enter your TOEFL score").grid(row=2)
Label(master,text= "Enter university rating").grid(row=3)
Label(master,text= "Enter SOP").grid(row=4)
Label(master,text= "Enter LOR").grid(row=5)
Label(master,text= "Enter your CGPA").grid(row=6)
Label(master,text= "Research").grid(row=7)


e1= Entry(master)
e2= Entry(master)
e3= Entry(master)
e4= Entry(master)
e5= Entry(master)
e6= Entry(master)
e7= Entry(master)

e1.grid(row=1,column=1)
e2.grid(row=2,column=1)
e3.grid(row=3,column=1)
e4.grid(row=4,column=1)
e5.grid(row=5,column=1)
e6.grid(row=6,column=1)
e7.grid(row=7,column=1)

Button(master,text="Predict",command=show_entry).grid()

mainloop()


# In[ ]:




