#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df= pd.read_csv("C:/Users/vaish/Downloads/Mall_Customers.csv")


# In[3]:


df.head()


# # univariate analysis

# In[4]:


df.describe()


# In[5]:


sns.distplot(df['Annual Income (k$)'])


# In[6]:


df.columns


# In[7]:


columns =['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns :
    plt.figure()
    sns.distplot(df[i])
    


# In[8]:


sns.kdeplot(df['Annual Income (k$)'],shade=True);


# In[9]:


columns= ['Age', 'Annual Income (k$)','Spending Score (1-100)']
for i in columns:
    plt.figure()
    sns.boxplot(data=df,x='Gender',y=df[i])


# In[10]:


df['Gender'].value_counts(normalize=True)


# # Bivariate Analysis

# In[11]:


sns.scatterplot(data=df,x='Annual Income (k$)', y='Spending Score (1-100)')


# In[12]:


df=df.drop('CustomerID',axis=1)
sns.pairplot(df)


# In[13]:


df.groupby(['Gender'])['Age', 'Annual Income (k$)','Spending Score (1-100)'].mean()


# In[14]:


df.corr()


# In[15]:


sns.heatmap(df.corr(),annot=True,cmap='coolwarm')


# 
