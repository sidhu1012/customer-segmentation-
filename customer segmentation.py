#!/usr/bin/env python
# coding: utf-8

# In[8]:


import random 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
from sklearn.cluster import KMeans 
from sklearn.datasets.samples_generator import make_blobs 
from sklearn.preprocessing import StandardScaler
get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


cust_df=pd.read_csv('https://s3-api.us-geo.objectstorage.softlayer.net/cf-courses-data/CognitiveClass/ML0101ENv3/labs/Cust_Segmentation.csv')


# In[7]:


df = cust_df.drop('Address', axis=1)
df.head()


# In[11]:


X = df.values[:,1:]
X = np.nan_to_num(X)
Clus_dataSet = StandardScaler().fit_transform(X)
Clus_dataSet


# In[13]:


clusterNum = 5
k_means = KMeans(init = "k-means++", n_clusters = clusterNum, n_init = 15)
k_means.fit(X)
labels = k_means.labels_
print(labels)


# In[14]:


df["Clus_km"] = labels
df.head(5)


# In[15]:


area = np.pi * ( X[:, 1])**2  
plt.scatter(X[:, 0], X[:, 3], s=area, c=labels.astype(np.float), alpha=0.5)
plt.xlabel('Age', fontsize=18)
plt.ylabel('Income', fontsize=16)


# In[18]:


from mpl_toolkits.mplot3d import Axes3D 
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, .95, 1], elev=48, azim=134)

plt.cla()
ax.set_ylabel('Age', fontsize=18)
ax.set_xlabel('Income', fontsize=16)
ax.set_zlabel('Education', fontsize=16)

ax.scatter(X[:, 3], X[:, 1], X[:, 0], c= labels.astype(np.float))


# In[ ]:




