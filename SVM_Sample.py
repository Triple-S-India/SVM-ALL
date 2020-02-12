#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[4]:


from sklearn.datasets.samples_generator import make_blobs


# In[23]:


X,y = make_blobs(n_samples = 100, centers = 2, random_state = 0, cluster_std = 0.60)


# In[24]:


plt.scatter(X[:,0],X[:,1], c= y, s = 50, cmap = 'autumn')
plt.show()


# In[25]:


xfit = np.linspace(-1,3.5)


# In[35]:


plt.scatter(X[:,0],X[:,1], c=y, s = 50,cmap='autumn')
plt.plot([0.6],[2.1],'x',color='red', markeredgewidth=2,markersize=10)
for m,b in [(0.80,1.30),(0.5,1.6),(-0.2,2.9)]:
    plt.plot(xfit,m*xfit+b, '-k')
plt.xlim(-1,3.5)
plt.show()


# In[38]:


plt.scatter(X[:,0],X[:,1], c=y, s = 50,cmap='autumn')
plt.plot([0.6],[2.1],'x',color='red', markeredgewidth=2,markersize=10)
for m,b,d in [(0.80,1.30,0.35),(0.5,1.6,0.55),(-0.2,2.9,0.2)]:
    yfit = xfit*m+b
    plt.plot(xfit,yfit,'-k')
    plt.fill_between(xfit,yfit-d,yfit+d,edgecolor='none',color='blue',alpha=0.4)
plt.xlim(-1,3.5)


# In[39]:


from sklearn.svm import SVC       #Support_Vector_Classifier


# In[42]:


model = SVC(kernel='linear', C = 1E10)
model.fit(X,y)


# In[43]:


def plot_svc_decision_function(model,ax=None,plot_support= True):
    #Plot a decision function for a 2d SVC
    if ax is None:
        ax = plt.gca()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    
    #Create grid to evaluate model
    x = np.linspace(xlim[0],xlim[1],30)
    y = np.linspace(ylim[0],ylim[1],30)
    Y,X = np.meshgrid(y,x)
    xy = np.vstack([X.ravel(),Y.ravel()]).T
    P = model.decision_function(xy).reshape(X.shape)
    
    #Plot decision boundary and margins
    ax.contour(X,Y,P, colors = 'k', levels = [-1,0,1], alpha = 0.5, linestyle =['--','-','--'])
    
    #Plot support vectors
    if plot_support:
        ax.scatter(model.support_vectors_[:,0], model.support_vectors_[:,1], s = 300, linewidth = 1, facecolor = 'none')
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
plt.scatter(X[:,0],X[:,1], c = y, s = 50, cmap = 'autumn')
plot_svc_decision_function(model)


# In[47]:


pred = model.predict(X)


# In[50]:


from sklearn.metrics import classification_report, confusion_matrix
print(confusion_matrix(y,pred))


# In[51]:


print(classification_report(y, pred))


# In[ ]:




