# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 15:22:03 2023

@author: icon
"""

import pandas as pd
from numpy import array
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import svd
A=array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])

print(A)
#SVD
U,d,Vt=svd(A)
print(U)
print(d)
print(Vt)
print(np.diag(d))

data=pd.read_excel("C:/University_Clustering.xlsx")
data.head()
data=data.iloc[:,2:]# non numeric data
data
from sklearn.decomposition import TruncatedSVD
svd= TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns="pc0","pc1","pc3"
result.head()
#scatter diagram
import matplotlib.pyplot as plt
plt.scatter(x=result.pc0,y=result.pc1)








