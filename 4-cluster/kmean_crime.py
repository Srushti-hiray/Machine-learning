# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:50:42 2023

@author: icon
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:18:00 2023

@author: icon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

#let us first understahd how kmeans work for  2 dimensional data set
#for taht generate random numbers from 0 to 1
# in uniform probablity of 1/50

X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create empty daatframe with 0 rows and 2 columns

df_xy=pd.DataFrame(columns=["X","Y"])
#assign value of X,Y to this columns

df_xy.X=X
df_xy.Y=Y
model1=KMeans(n_clusters=3).fit(df_xy)

"""
with data X and Y apply kmean model.generate scatter plot with scale /font=10
cmap=plt.cm.coolwarm:cool color combination
"""
model1.labels_
df_xy.plot(x="X",y="Y",c=model1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

crime=pd.read_csv("C:/CSV files/crime_data.csv")
cr=crime.describe()
crime.columns
crime.drop(['Unnamed: 0'],axis=1,inplace=True)

#as scale of all quantitive data is more so either standardize it or normalize  it
#whenever there is mixed data use normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now apply this normalization function to uninew dataframe for all the rows
#since 0Th column has univercity name hence skipp it
df_norm=norm_func(crime.iloc[:,:])

#noe df_norm dataframe is scaled DF
#now you can apply describe
b=df_norm.describe()

"""
what will be ideal culster number 1,2 or 3
"""

TWSS=[]
k=list(range(2,8))
for i in k:
    kmeans=KMeans(n_clusters=i)
    kmeans.fit(df_norm)
    TWSS.append(kmeans.inertia_)#total with in sum of squares
    
    """
    KMeans inertia also known as sum of squares errors or SSE , calculates sum
    of distances of all the points with in the cluster from centroid of the points
    it is difference between the observed value and predicted value

    
    """
TWSS
#the k value increases the TWSS value decreses

plt.plot(k,TWSS,"ro-");
plt.xlabel("No of clusters");
plt.ylabel("Total_within_ss")        

"""
how to select value of k with in the elbow curve
when k changes fron 2 to 3,then desreases in twss in higher than 
when k changes fron 3 to 4 when k value changes from 5 to 6 in twss is constantly
less, hence considered k=3
"""

model1=KMeans(n_clusters=3)
model1.fit(df_norm)
model1.labels_
mb=pd.Series(model1.labels_)


df_norm.insert(0,"clust",mb)#use other method to reset index
df_norm
df_norm.iloc[:,2:].groupby(df_norm.clust).mean()
df_norm.to_csv("kmeans_crime.csv",encoding="utf-8")
import os
os.getcwd()







