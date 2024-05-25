# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:12:35 2023

@author: icon
"""
#if data set is small than agglomorative clustring
import pandas as pd
import matplotlib.pyplot as plt
insu=pd.read_csv("C:/CSV files/autoInsurance.csv")
a=insu.describe()
a.columns
insu1=insu[['Customer Lifetime Value', 'Income', 'Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies',
       'Total Claim Amount','Response', 'Coverage',
              'Education','EmploymentStatus','Gender','Marital Status','Policy Type','Renew Offer Type', 'Sales Channel','Vehicle Class', 'Vehicle Size']]

print(type(insu1))
insu1.dtypes

#1st identify duplicates

duplicate=insu1.duplicated()
sum(duplicate)
#so total 861 duplicates are present
insu2=insu1.drop_duplicates()
#now insu2 have no duplicates in it
a=insu2.describe()
a.columns

insu2.rename({'Customer Lifetime Value':'Customer_Lifetime_Value','Monthly Premium Auto':'Monthly_Premium_Auto','Months Since Last Claim':'Months_Since_Last_Claim','Months Since Policy Inception':'Months_Since_Policy_Inception','Number of Open Complaints':'Number_of_Open_Complaints','Number of Policies':'Number_of_Policies','Total Claim Amount':'Total_Claim_Amount','Marital Status':'Marital_Status','Policy Type':'Policy_Type','Renew Offer Type':'Renew_Offer_Type','Sales Channel':'Sales_Channel','Vehicle Class':'Vehicle_Class','Vehicle Size':'Vehicle_Size'},axis=1,inplace=True)

insu2.isnull().sum()
#no null values
insu2.columns
##################################################################

### now we will check if it have outlier
insu2.dtypes
import seaborn as sns
sns.boxplot(insu2.Income)#no outlier
sns.boxplot(insu2.Customer_Lifetime_Value)#have outlier
sns.boxplot(insu2.Months_Since_Last_Claim)#no outlier
sns.boxplot(insu2.Months_Since_Policy_Inception)#no outlier
sns.boxplot(insu2.Number_of_Open_Complaints)
sns.boxplot(insu2.Number_of_Policies)#have outlier
sns.boxplot(insu2.Total_Claim_Amount)#have outlier

##################################################################

#Winsorizer
#removing outliers

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method="iqr",
                  tail="both",
                  fold=1.5,
                  variables=["Customer_Lifetime_Value",'Number_of_Policies','Total_Claim_Amount'])
insu_new=winsor.fit_transform(insu2[["Customer_Lifetime_Value",'Number_of_Policies','Total_Claim_Amount']])

#now check if there is outlier or not

sns.boxplot(insu_new.Customer_Lifetime_Value)
sns.boxplot(insu_new.Number_of_Policies)
sns.boxplot(insu_new.Total_Claim_Amount)
insu_new.columns
# there is no outlier

#################################################################

# now combine this "Customer_Lifetime_Value",'Number_of_Policies','Total_Claim_Amount'
# column from insu_new to the remaining from insu2

insu2.drop(columns=['Customer_Lifetime_Value','Number_of_Policies','Total_Claim_Amount'],inplace=True)
insu2.columns

insu2['Customer_Lifetime_Value']=insu_new['Customer_Lifetime_Value']
insu2['Number_of_Policies']=insu_new['Number_of_Policies']
insu2['Total_Claim_Amount']=insu_new['Total_Claim_Amount']

# check if now outlier of column 'Customer_Lifetime_Value' so that 
#we will be sure that this column is proper added
sns.boxplot(insu2.Customer_Lifetime_Value)

###################################################################

### now check datatypes of qualitative data 
#clasify it as ordinal and nominal
insu2.dtypes

######## by using value_counts() we get cagerory of varibale

insu2['Coverage'].value_counts()
insu2['Education'].value_counts()
insu2['EmploymentStatus'].value_counts()
insu2['Gender'].value_counts()
insu2['Marital_Status'].value_counts()
insu2['Policy_Type'].value_counts()
insu2['Renew_Offer_Type'].value_counts()
insu2['Sales_Channel'].value_counts()
insu2['Vehicle_Class'].value_counts()
insu2['Vehicle_Size'].value_counts()

"""
Response ="yes","no"  2 :M=1,Y=0
Coverage="basic","extended","premium"   3
Education=Bachelor ,College ,High School or Below ,Master ,Doctor   5 
EmploymentStatus=5
Gender=2 : F=1,M=0
Marital_Status=3
Policy_Type=3
Renew_Offer_Type=4
Sales_Channel=4
Vehicle_Class=6
Vehicle_Size=3


All are nominal data so generate dummies

"""
insu2_new=pd.get_dummies(insu2)

# Response and Gender have 2 categories so remove one dummy
insu2_new.drop(["Response_No","Gender_F"],axis=1,inplace=True)

##########################################################

#now normalize the data to make it in one scale

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

insu_norm=norm_fun(insu2_new)
a=insu_norm.describe()

############################################################
#dandrogram

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(insu_norm,method="complete",metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("danddrogram")
plt.xlabel("index")
plt.ylabel("distance")
sch.dendrogram(z,leaf_rotation=0,leaf_font_size=10);
plt.show()

from sklearn.cluster import AgglomerativeClustering
insurance=AgglomerativeClustering(n_clusters=3,linkage="complete",affinity="euclidean").fit(insu_norm)
#generate labels 
insurance.labels_
cluster_labels=pd.Series(insurance.labels_)

insu_norm["clust"]=cluster_labels

insu_norm.drop(["clust"],axis=1,inplace=True)#dropping bcoz we will 
#use other method to reset index

insu_norm.insert(0,"clust",cluster_labels) # reset cluster to index 0
insu_norm["clust"].value_counts()

insu_norm.to_csv("autoInsurance.csv",encoding="utf-8")
import os
os.getcwd()

