# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 20:42:31 2023

@author: icon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

crime=pd.read_csv("C:/CSV files/crime_data.csv")

crime.shape
#(50, 5)

crime.columns
"""
['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape']
"""
duplicate=crime.duplicated()
sum(duplicate)
#no duplicates

crime.isnull().sum()

sns.boxplot(crime['Murder'])# no outlier
sns.boxplot(crime['Assault'])# no outlier
sns.boxplot(crime['UrbanPop'])# no outlier
sns.boxplot(crime['Rape'])# there is outlier

###############################################################

#Winsorizer
#removing outliers

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",
                  tail="both",
                  fold=1.5,
                  variables=["Rape"])
crime_new=winsor.fit_transform(crime[["Rape"]])

#now check if there is outlier or not

sns.boxplot(crime_new.Rape) # no outlier

crime.drop(columns=["Rape"])
crime["Rape"]=crime_new["Rape"]

################################################################
#normalize

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

crime_norm=norm_fun(crime.iloc[:,1:])
a=crime_norm.describe()

#################################################################

#dandrogram

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(crime_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("danddrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10);
plt.show()

from sklearn.cluster import AgglomerativeClustering
crime_data=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(crime_norm)
#generate labels 
crime_data.labels_
cluster_labels=pd.Series(crime_data.labels_)


crime_norm.insert(0,"clust",cluster_labels) # reset cluster to index 0
crime_norm["clust"].value_counts()

crime_norm.to_csv("crime_data.csv",encoding="utf-8")
import os
os.getcwd()

























