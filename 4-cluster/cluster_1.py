# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:12:35 2023

@author: icon
"""
#if data set is small than agglomorative clustring
import pandas as pd
import matplotlib.pyplot as plt
uni=pd.read_excel("C:/CSV files/University_Clustering.xlsx")

a=uni.describe()
#column state will not affect the data so drop it
uninew=uni.drop(["State"],axis=1)
#as scale of all quantitive data is more so either standardize it or normalize  it
#whenever there is mixed data use normalization
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x
#now apply this normalization function to uninew dataframe for all the rows
#since 0Th column has univercity name hence skipp it
df_norm=norm_func(uninew.iloc[:,1:])
#noe df_norm dataframe is scaled DF
#now you can apply describe
b=df_norm.describe()


#before applying clustring you need to apply dandrograme first
#now to craete dandrogram we need to measure distance
from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
# linkage function gives you hierarchical or agglomerative clustring
z=linkage(df_norm,method="complete",metric='euclidean')
plt.figure(figsize=(15,8));
plt.title("hierarchical clustering dandrogram");
plt.xlabel("Index");
plt.ylabel("Distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10);
plt.show()
#dandrogram
#apply the agglomerative clustring choosing 3 as cluster 
#from dandrograme
#whatever has been dislay in dandrogram is not clustering
#it is just showing number of possible clustering
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(df_norm)

#apply labels to cluster
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to uninew DF as column and name column as cluster
uni['clust']=cluster_labels
#we want to relocate column clust to 1st column
#         uninew=uni.iloc[:,[7,1,2,3,4,5,6]]
#noe check uninew DF
uni.iloc[:,2:].groupby(uni.clust).mean()

uni=uni.iloc[:,[7,1,2,3,4,5,6]]
#from output cluster 2 has got highest top 10
#lowest accept ratio,best faculty ratio and highest expenses
#highest graduate ratio
uni.to_csv("University.csv",encoding="utf-8")
import os
os.getcwd()












