# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 06:38:25 2023

@author: icon
"""

import pandas as pd 
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

df=pd.read_csv("C:/CSV files/Telco_customer_churn.csv")
df.info()

"""
"Quarter","Referred a Friend","Offer","Phone Service","Multiple Lines",
"Internet Service","Internet Type","Online Security","Online Backup","Device Protection Plan",
"Premium Tech Support","Streaming TV","Streaming Movies","Streaming Music","Unlimited Data",
"Contract","Paperless Billing","Payment Method"

"""
sns.boxplot(df["Tenure in Months"])#no outlier
sns.boxplot(df["Avg Monthly Long Distance Charges"])#no outlier
sns.boxplot(df["Monthly Charge"])# no outlier
sns.boxplot(df["Total Charges"])# no outlier
sns.boxplot(df["Total Extra Data Charges"])
sns.boxplot(df["Total Revenue"])# have outlier

##################################################################

#Winsorizer
#removing outliers

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",
                  tail="both",
                  fold=1.5,
                  variables=["Total Revenue"])
tele_new=winsor.fit_transform(df[["Total Revenue"]])

sns.boxplot(tele_new["Total Revenue"])

df.drop(["Total Revenue"],axis=1,inplace=True)
df["Total Revenue"]=tele_new["Total Revenue"]

#############################################################

#create dummies
df.drop(["Customer ID"],axis=1,inplace=True)
df_new=pd.get_dummies(df)

##########################################################

#now normalize the data to make it in one scale

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(df_new)
a=df_norm.describe()
df_norm.drop(["Count","Quarter_Q3"],axis=1,inplace=True)
############################################################

#dandrogram

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("danddrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10);
plt.show()

from sklearn.cluster import AgglomerativeClustering
teleco=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)
#generate labels 
teleco.labels_
cluster_labels=pd.Series(teleco.labels_)

df_norm.insert(0,"clust",cluster_labels) # reset cluster to index 0
df_norm["clust"].value_counts()

df_norm.to_csv("Telco_customer_churn.csv",encoding="utf-8")
import os
os.getcwd()
















