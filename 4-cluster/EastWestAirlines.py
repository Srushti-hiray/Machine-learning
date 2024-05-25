# -*- coding: utf-8 -*-
"""
Created on Mon Oct 16 07:24:31 2023

@author: icon
"""

########################## EastWestAirlines #########################

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

air=pd.read_csv("C:/CSV files/EastWestAirlines.csv")
#
air.shape
#(3999,12)
air.columns
"""
['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award?']
"""
# all the columns are numeric data type

#now check if it have missing values or not
air.isnull().sum()
#there is no missing values in any column

#now check duplicates
duplicate=air.duplicated()
sum(duplicate)
#their is no duplicates

air.columns
"""
['ID#', 'Balance', 'Qual_miles', 'cc1_miles', 'cc2_miles', 'cc3_miles',
       'Bonus_miles', 'Bonus_trans', 'Flight_miles_12mo', 'Flight_trans_12',
       'Days_since_enroll', 'Award?']
"""
#now checking outliers
sns.boxplot(air["ID#"]) #no outliers
sns.boxplot(air["Balance"]) # have outliers
sns.boxplot(air["Qual_miles"]) #have outliers
sns.boxplot(air["cc1_miles"]) # no outliers
sns.boxplot(air["cc2_miles"])
sns.boxplot(air["cc3_miles"])
sns.boxplot(air["Bonus_miles"]) #have outliers
sns.boxplot(air["Bonus_trans"]) #have outliers
sns.boxplot(air["Flight_miles_12mo"]) #have outliers
sns.boxplot(air["Flight_trans_12"]) #have outliers
sns.boxplot(air["Days_since_enroll"]) #no outliers
sns.boxplot(air["Award?"]) # no outliers


#########################################################

#winsorizer

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",
                  tail="both",
                  fold=1.5,
                  variables=["Balance","Bonus_miles","Bonus_trans","Flight_miles_12mo","Flight_trans_12"])
air_new=winsor.fit_transform(air[["Balance","Bonus_miles","Bonus_trans","Flight_miles_12mo","Flight_trans_12"]])

# "Qual_miles" :Input columns ['Qual_miles'] have low variation for method 'iqr'. Try other capping methods or drop these columns.

sns.boxplot(air_new["Balance"]) 
sns.boxplot(air_new["Bonus_miles"]) 
sns.boxplot(air_new["Bonus_trans"]) 
sns.boxplot(air_new["Flight_miles_12mo"])
sns.boxplot(air_new["Flight_trans_12"])

air.drop(columns=["Balance","Bonus_miles","Bonus_trans","Flight_miles_12mo","Flight_trans_12","Qual_miles"],inplace=True)
air["Balance"]=air_new["Balance"]
air["Bonus_miles"]=air_new["Bonus_miles"]
air["Bonus_trans"]=air_new["Bonus_trans"]
air["Flight_miles_12mo"]=air_new["Flight_miles_12mo"]
air["Flight_trans_12"]=air_new["Flight_trans_12"]

############################################################

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

air_norm=norm_fun(air)
a=air_norm.describe()

############################################################
#dandrogram

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(air_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("danddrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10);
plt.show()

from sklearn.cluster import AgglomerativeClustering
airlines=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(air_norm)
#generate labels 
airlines.labels_
cluster_labels=pd.Series(airlines.labels_)

air_norm["clust"]=cluster_labels

air_norm.drop(["clust"],axis=1,inplace=True)#dropping bcoz we will 
#use other method to reset index

air_norm.insert(0,"clust",cluster_labels) # reset cluster to index 0
air_norm["clust"].value_counts()

air_norm.to_csv("EastWestAirlines.csv",encoding="utf-8")
import os
os.getcwd()





























