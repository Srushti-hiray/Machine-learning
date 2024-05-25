# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 16:37:56 2023

@author: icon
"""

import pandas as pd
import numpy as np

uni1=pd.read_csv("C:/4-cluster/University.csv")
uni1.describe()
uni1.info()
uni=uni1.drop(["State"],axis=1)

from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale

#considering only numeric data
uni.data=uni.iloc[:,1:]

#scalable/normalize the data
uni_norm1=scale(uni.data)
uni_norm1

pca=PCA(n_components=6)
pca_values=pca.fit_transform(uni_norm1)

#the amount of variance that ecah PCA explains is
var=pca.explained_variance_ratio_
var

#PCA weights
#pca.components_
#pca.components[0]

#cummulative variance
var1=np.cumsum(np.round(var,decimals=4)*100)
var1

#variance plot for PCA components obtained
plt.plot(var1,color="red")

#PCA score
pca_values

pca_data=pd.DataFrame(pca_values)
pca_data=columns="comp0","comp1","comp2","comp3","comp4","comp5"
final=pd.concat([uni.Univ,pca_data.iloc[:,0:3]],axis=1)
#this is "Univ" columns of uni data frame

#scatter diagram

import matplotlib.pyplot as plt
ax=final.plot(x="comp0",y="comp1",kind="scatter",figsize=(12,8))
final[["comp0","comp1","Univ"]].apply(lambda x:ax.test(*x),axis=1)























