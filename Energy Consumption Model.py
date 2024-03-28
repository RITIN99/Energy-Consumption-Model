#!/usr/bin/env python
# coding: utf-8

# In[35]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
data1 = pd.read_csv('2010-2011 Solar home electricity data.csv')
data2 = pd.read_csv('2011-2012 Solar home electricity data v2.csv')
data3 = pd.read_csv('2012-2013 Solar home electricity data v2.csv')
md = pd.concat([data1,data2,data3])
print(md.head())


# In[36]:


md.shape


# In[37]:


md.info


# In[38]:


del md['Row Quality']


# In[39]:


md.info()


# In[40]:


md.isnull().sum()


# In[41]:


avg=md.iloc[:,5:].mean(axis=1)


# In[42]:


md["Average"]=avg


# In[43]:


date_components = md["Date"].str.split("/", expand=True)

md["Day"] = date_components[0].astype(int)
md["Month"] = date_components[1].astype(int)
md["Year"] = date_components[2].astype(int)


# In[44]:


md.head()


# In[45]:


dummy=pd.get_dummies(md['Consumption Category'],dtype=int)
dummy.columns


# In[46]:


md=pd.concat([md,dummy],axis=1)


# In[47]:


md.head()


# In[48]:


md_cl=md[md.CL==1]


# In[49]:


md_cl


# In[50]:


md_gc=md[md.GC==1]


# In[51]:


md_gc


# In[52]:


md_gg=md[md.GG==1]


# In[53]:


md_gg


# In[54]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
z=le.fit_transform(md_cl.Month)


# In[55]:


cl=LinearRegression()
gg=LinearRegression()
gc=LinearRegression()
models=[cl,gg,gc]


# In[56]:


X_cl=md_cl.loc[:,['Generator Capacity','Postcode','Month']]
Y_cl=md_cl.Average


# In[57]:


X_gc=md_gc.loc[:,['Generator Capacity','Postcode','Month']]
Y_gc=md_gc.Average


# In[58]:


X_gg=md_gg.loc[:,['Generator Capacity','Postcode','Month']]
Y_gg=md_gg.Average


# In[59]:


from sklearn.model_selection import train_test_split
X=[X_cl,X_gg,X_gc]
Y=[Y_cl,Y_gg,Y_gc]


# In[60]:


train=[]
test=[]
for i in range(len(X)):
    train_x,test_x,train_y,test_y=train_test_split(X[i],Y[i],train_size=0.80,random_state=4)
    train.append((train_x,train_y))
    test.append((test_x,test_y))


# In[61]:


for i in range(3):
    models[i].fit(train[i][0],train[i][1])
    print(i,models[i].score(train[i][0],train[i][1]))


# In[62]:


for i in range(3):
    print(i,models[i].score(test[i][0],test[i][1]))


# In[63]:


from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error

for i in range(3):
    predictions = models[i].predict(test[i][0])

    rmse = np.sqrt(mean_squared_error(test[i][1], predictions))
    mse = mean_squared_error(test[i][1], predictions)
    mae = mean_absolute_error(test[i][1], predictions)
    mape = mean_absolute_percentage_error(test[i][1], predictions)
    
    print(i, rmse, mse, mae, mape)


# In[ ]:




