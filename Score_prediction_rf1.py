#!/usr/bin/env python
# coding: utf-8

# In[4]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:



# In[2]:

def predict(number1,number2,number3,number4,number5,number6):


# Importing the dataset
    import pandas as pd
    if(number6=="ODI"):
        dataset = pd.read_csv("odi.csv")
    elif(number6=="IPL"):
        dataset=pd.read_csv("ipl.csv")
    X = dataset.iloc[:,[7,8,9,12,13]].values
    y = dataset.iloc[:, 14].values


# In[ ]:





# In[3]:



# Splitting the dataset into the Training set and Test set
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)


# In[4]:


# Feature Scaling
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)


# In[5]:


# Training the dataset
    from sklearn.ensemble import RandomForestRegressor
    reg = RandomForestRegressor(n_estimators=70,max_features=None)
    reg.fit(X_train,y_train)# Testing the dataset on trained model



# In[6]:


    y_pred = reg.predict(X_test)


# In[7]:


# Testing with a custom input

    import numpy as np
    new_prediction = reg.predict(sc.transform(np.array([[250,4,45,100,50]])))


# In[8]:


    import numpy as np
    result = reg.predict(sc.transform(np.array([[number1,number2,number3,number4,number5]])))
    return result[0]



# In[9]:


# In[ ]:




