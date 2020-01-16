#!/usr/bin/env python
# coding: utf-8

# In[43]:


import pandas as pd  # pandas for data frame 
import numpy as np  # numpy 
import matplotlib.pyplot as plt # for graphs 
import seaborn as sns # for graphs 
import pprint
import time  # for day light savings 


# In[44]:


from sklearn.model_selection import train_test_split 
from sklearn.pipeline import Pipeline # for creating a pipeline 
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler # for standardizing data 
from sklearn.svm import SVC # support vector machines 
from sklearn.model_selection import GridSearchCV


# In[45]:


startT = time.time()


winedf = pd.read_csv('winequality-red.csv',sep=';') # reading the file into a dataframe using pandas library 


# In[46]:


winedf #viewing the file as such ! 


# In[47]:


winedf.head(3) # first 3 rows 


# In[48]:


winedf.isnull().sum() # there are no missing values in the DF 


# In[49]:


winedf.shape


# In[50]:


winedf[["quality"]]


# In[51]:


winecorr = winedf.corr()


# In[52]:


s=sns.heatmap(winecorr)
s.set_yticklabels(s.get_yticklabels(),rotation=30,fontsize=10)
s.set_xticklabels(s.get_xticklabels(),rotation=30,fontsize=10)

plt.show() # as expected high correlation between acidity and pH


# In[53]:


# individual correlation plot

# first plot is between fixed acidity and ph and measured by quality 
plt.subplot(1,2,1)
plt.scatter(winedf['fixed acidity'], winedf['pH'], s=winedf['quality']*5, color='magenta', alpha=0.3)
plt.xlabel('Fixed Acidity')
plt.ylabel('pH')
plt.subplot(1,2,2)

# second plot is between fixed acidity and residual sugar and measured by quality 
plt.scatter(winedf['fixed acidity'], winedf['residual sugar'], s=winedf['quality']*5, color='purple', alpha=0.3)
plt.xlabel('Fixed Acidity')
plt.ylabel('Residual Sugar')
plt.tight_layout()
plt.show()


# In[54]:


# In this data frame we will choose the quality to be the target, 

# in the following code I am splitting up the data frame such that the main data frame does not have the quality column - so we 
# effectively drop it 

# and secondly a target variable is created where we set quality to be the target variable. 

X=winedf.drop(['quality'],axis=1)

Y=winedf['quality'] 


# In[55]:


X  # we can see in this quality has been dropped. 


# In[56]:


Y


# In[57]:


type(X)


# In[58]:


type(Y)


# In[59]:


X.head(3)


# In[60]:


#++++++++++++++++++++++++++++++++
# create the pipeline object
#++++++++++++++++++++++++++++++++
steps = [('scaler', StandardScaler()), ('SVM', SVC())]
pipeline = Pipeline(steps)


# In[61]:


pipeline


# In[62]:


parameteres = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01]}


# In[63]:


parameteres


# In[64]:


X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size=0.2, random_state=30, stratify=Y)


# In[66]:


grid = GridSearchCV(pipeline, param_grid=parameteres, cv=5)

grid.fit(X_train, y_train)


# In[73]:


(grid.score(X_test,y_test))


# In[74]:


grid.best_params_


# In[77]:


(endT-startT)


# In[75]:


endT = time.time()


# In[76]:


endT


# In[ ]:




