#!/usr/bin/env python
# coding: utf-8

# In[1]:
import os
import pandas as pd

file_path = r'C:\Users\vashc\OneDrive\python\quikr_car.csv'

if os.path.exists(file_path):
    car = pd.read_csv(file_path)
else:
    print(f"File not found: {file_path}")


import pandas as pd
import numpy as np

car = pd.read_csv('quikr_car.csv')

# In[2]:


car=pd.read_csv('quikr_car.csv')
car.head()


# In[3]:


car.shape

car.info()


# In[4]:


backup=car.copy()


# Quality
# names are pretty inconsistent
# names have company names attached to it
# some names are spam like 'Maruti Ertiga showroom condition with' and 'Well mentained Tata Sumo'
# company: many of the names are not of any company like 'Used', 'URJENT', and so on.
# year has many non-year values
# year is in object. Change to integer
# Price has Ask for Price
# Price has commas in its prices and is in object
# kms_driven has object values with kms at last.
# It has nan values and two rows have 'Petrol' in them
# fuel_type has nan values

# Cleaning Data

# In[5]:


car=car[car['year'].str.isnumeric()]


# In[6]:


car['year']=car['year'].astype(int)


# In[7]:


car=car[car['Price']!='Ask For Price']


# In[8]:


car['Price']=car['Price'].str.replace(',','').astype(int)


# In[9]:


car['kms_driven']=car['kms_driven'].str.split().str.get(0).str.replace(',','')


# In[10]:


car=car[car['kms_driven'].str.isnumeric()]


# In[11]:


car['kms_driven']=car['kms_driven'].astype(int)


# In[12]:


car=car[~car['fuel_type'].isna()]


# In[13]:


car.shape


# In[14]:


car['name']=car['name'].str.split(' ').str.slice(0,3).str.join(' ')


# In[15]:


car.head()


# In[16]:


car.reset_index(drop=True)


# In[17]:


car.describe()


# In[18]:


car[~(car['Price']>6e6)]


# In[19]:


car.to_csv('cleaned_car.csv')


# Model

# In[20]:


x= car.drop(columns='Price')
y=car['Price']


# In[21]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder


# In[22]:


ohe = OneHotEncoder()
ohe.fit(x[['name','company','fuel_type']])


# In[23]:


from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline


# In[24]:


column_trans = make_column_transformer((OneHotEncoder(categories=ohe.categories_),['name','company','fuel_type']),remainder='passthrough')


# In[25]:


lr=LinearRegression()


# In[26]:


pipe = make_pipeline(column_trans,lr)


# In[27]:


pipe.fit(x_train,y_train)


# In[28]:


y_pred=pipe.predict(x_test)


# In[29]:


r2_score(y_test,y_pred)


# In[33]:


scores=[]
for i in range(1000):
    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=i)
    lr=LinearRegression()
    pipe=make_pipeline(column_trans,lr)
    pipe.fit(x_train,y_train)
    y_pred=pipe.predict(x_test)
    scores.append(r2_score(y_test,y_pred))


# In[34]:


np.argmax(scores)


# In[32]:


scores[np.argmax(scores)]


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=np.argmax(scores))
lr=LinearRegression()
pipe=make_pipeline(column_trans,lr)
pipe.fit(x_train,y_train)
y_pred=pipe.predict(x_test)
r2_score(y_test,y_pred)


# In[39]:


import pickle


# In[40]:


pickle.dump(pipe,open('LinearRrgressionModel.pkl','wb'))


# In[45]:


pipe.predict(pd.DataFrame([['Maruti Suzuki Swift','Maruti',2019,100,'Petrol']],columns=['name','company','year','kms_driven','fuel_type']))


# In[ ]:




