# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 16:04:42 2023

@author: icon
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 08:26:03 2023

@author: icon
"""

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
a.columns

             
print(type(uni))
uni.dtypes

#1st identify duplicates

duplicate=uni.duplicated()
sum(duplicate)
#so no duplicates are present

a.columns

##################################################################

### now we will check if it have outlier
uni.dtypes
import seaborn as sns
sns.boxplot(uni.SAT)#have outlier
sns.boxplot(uni.Top10)#have outlier
sns.boxplot(uni.Accept)#have outlier
sns.boxplot(uni.SFRatio)#have outlier
sns.boxplot(uni.Expenses)#no outlier
sns.boxplot(uni.GradRate)#no outlier


##################################################################

#Winsorizer
#removing outliers

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",
                  tail="both",
                  fold=1.5,
                  variables=['SAT','Top10','Accept','SFRatio'])
uni_new=winsor.fit_transform(uni[['SAT','Top10','Accept','SFRatio']])

#now check if there is outlier or not

sns.boxplot(uni_new.SAT)

# there is no outlier

#################################################################

# now combine this 'Expenses','GradRate'
# column from uni_new to the remaining from uni

uni.drop(columns=['SAT', 'Top10', 'Accept', 'SFRatio'],inplace=True)
uni.columns

uni['SAT']=uni_new["SAT"]
uni["Top10"]=uni_new["Top10"]
uni['Accept']=uni_new['SFRatio']

# check if now outlier of column 'SAT' so that 
#we will be sure that this column is proper added
sns.boxplot(uni.SAT)

###################################################################

uni.drop(['Univ', 'State'],axis=1,inplace=True)


##########################################################

#now normalize the data to make it in one scale

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

uni_norm=norm_fun(uni_new)
a=uni_norm.describe()

############################################################
#dandrogram

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(uni_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("danddrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10);
plt.show()

from sklearn.cluster import AgglomerativeClustering
univ=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(uni_norm)
#generate labels 
univ.labels_
cluster_labels=pd.Series(univ.labels_)

uni_norm.insert(0,"clust",cluster_labels) # reset cluster to index 0


uni_norm.to_csv("university.csv",encoding="utf-8")
import os
os.getcwd()

