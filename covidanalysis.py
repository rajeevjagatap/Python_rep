#!/usr/bin/env python
# coding: utf-8

# # COVID Analyis 

# In[33]:


import pandas as pd
import datetime as dt
from pandas import datetime
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sea
from sklearn import linear_model
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

rawdf = pd.read_excel('covidrawdata.xlsx', header=0, parse_dates=[0], index_col=0, squeeze=True,)
from sklearn.linear_model import LinearRegression

### Select required Fields
# In[3]:


rawdf1 = rawdf.loc[:,("Patient Number", "Merged")]#, "Week of Year")]
#rawdf1['Week of Year'] = pd.to_numeric(rawdf1['Week of Year'])
rawdf1


# In[ ]:





# In[65]:


rawdf1 = rawdf1.dropna()
rawdf1['Merged'] = rawdf1['Merged']
rawdf1['Merged'] = pd.to_datetime(rawdf1['Merged'])
rawdf1['Patient Number'] = rawdf1['Patient Number'].apply(np.int64)
rawdf1.info()


# In[6]:


plt.plot(rawdf1['Merged'], rawdf1['Patient Number'], color = 'red')
plt.xlabel("Week")
plt.ylabel("Patients")


# In[7]:


plt.bar(rawdf1['Merged'], rawdf1['Patient Number'])
plt.xlabel("Date")
plt.ylabel("Patients")


# In[19]:


plt.show()


# In[8]:


sea.distplot(rawdf1['Merged'])


# In[ ]:





# In[49]:


reg = linear_model.LinearRegression()
y = rawdf1['Patient Number'].values.reshape(-1,1)
X = rawdf1['Merged'].values.reshape(-1,1)


# In[68]:


print(reg)


# In[50]:


reg.fit(X,y)


# In[51]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.8, random_state=0)


# In[52]:


regressor = LinearRegression() 


# In[54]:


regressor.fit(X_train, y_train)


# In[55]:


print(regressor.intercept_)


# In[56]:


print(regressor.coef_)


# In[58]:


X_test


# In[78]:


import datetime as dt

X_test = pd.DataFrame(np.array(
    [['2020-04-03T00:00:00.000000000'],
       ['2020-04-07T00:00:00.000000000'],
       ['2020-04-06T00:00:00.000000000'],
       ...,
       ['2020-04-01T00:00:00.000000000'],
       ['2020-04-03T00:00:00.000000000'],
       ['2020-03-27T00:00:00.000000000']], dtype='datetime64[ns]'))

X_test.columns = ["Merged"]
X_test['Merged'] = pd.to_date(X_test['Merged'])
X_test['Merged'] = X_test['Merged'].map(dt.datetime.toordinal)


# In[77]:


y_pred = regressor.predict(X_test)


# In[ ]:


#df = pd.DataFrame({'Actual': y_test.flatten(), 'Predicted': y_pred.flatten()})
#df


# In[ ]:


#plt.scatter(X_test, y_test,  color='gray')
#plt.plot(X_test, y_pred, color='red', linewidth=2)
#plt.show()


# In[ ]:


#print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))  
#print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))  
#print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))


# In[ ]:




