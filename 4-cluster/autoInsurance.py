# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 16:39:18 2023

@author: icon
"""

import pandas as pd
import matplotlib.pyplot as plt
insu=pd.read_csv("C:/CSV files/autoInsurance.csv")

insu.columns
a=insu.describe()
a.columns
def norm_func(i):
    x=(i-i.min())/(i.max()-i.min())
    return x


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

df=insu2.rename({'Customer Lifetime Value':'Customer_Lifetime_Value','Monthly Premium Auto':'Monthly_Premium_Auto','Months Since Last Claim':'Months_Since_Last_Claim','Months Since Policy Inception':'Months_Since_Policy_Inception','Number of Open Complaints':'Number_of_Open_Complaints','Number of Policies':'Number_of_Policies','Total Claim Amount':'Total_Claim_Amount','Marital Status':'Marital_Status','Policy Type':'Policy_Type','Renew Offer Type':'Renew_Offer_Type','Sales Channel':'Sales_Channel','Vehicle Class':'Vehicle_Class','Vehicle Size':'Vehicle_Size'},axis=1,inplace=True)
df.isnull().sum()
#no null values

### now we will check if it have outlair
df.dtypes
import seaborn as sns
sns.boxplot(df.Income)#no outlier
sns.boxplot(df.Customer_Lifetime_Value)#have outlier
sns.boxplot(df.Months_Since_Last_Claim)#have outlier
sns.boxplot(df.Months_Since_Policy_Inception)#no outlier
sns.boxplot(df.Number_of_Open_Complaints)
sns.boxplot(df.Number_of_Policies)#have outlier
sns.boxplot(df.Total_Claim_Amount)#have outlier


#shape of insu1 is 9134,19 in which it consist of qualitative as well as quantitative data
#so apply dummies for that purpose to convert qualitative data into quantitative

import numpy as np
iqrC=df.Customer_Lifetime_Value.quantile(0.75)-df.Customer_Lifetime_Value.quantile(0.25)
LLC=df.Customer_Lifetime_Value.quantile(0.25)-iqrC*1.5
ULC=df.Customer_Lifetime_Value.quantile(0.75)+iqrC*1.5

df_Customer_Lifetime_Value=pd.DataFrame(np.where(df.Customer_Lifetime_Value>ULC,ULC,np.where(df.Customer_Lifetime_Value<LLC,LLC,df.Customer_Lifetime_Value)))
sns.boxplot(df_Customer_Lifetime_Value)# now there is no outlier
    
##########################

iqrM=df.Months_Since_Last_Claim.quantile(0.75)-df.Months_Since_Last_Claim.quantile(0.25)
LLM=df.Months_Since_Last_Claim.quantile(0.25)-iqrM*1.5
ULM=df.Months_Since_Last_Claim.quantile(0.75)+iqrM*1.5

df_Months_Since_Last_Claim=pd.DataFrame(np.where(df.Months_Since_Last_Claim>ULM,ULM,np.where(df.Months_Since_Last_Claim<LLM,LLM,df.Months_Since_Last_Claim)))
sns.boxplot(df_Months_Since_Last_Claim)

#########################

iqrN=df.Number_of_Policies.quantile(0.75)-df.Number_of_Policies.quantile(0.25)
LLN=df.Number_of_Policies.quantile(0.25)-iqrN*1.5
ULN=df.Number_of_Policies.quantile(0.75)+iqrN*1.5

df_Number_of_Policies=pd.DataFrame(np.where(df.Number_of_Policies>ULN,ULN,np.where(df.Number_of_Policies<LLN,LLN,df.Number_of_Policies)))
sns.boxplot(df_Number_of_Policies)

##########################

iqrT=df.Total_Claim_Amount.quantile(0.75)-df.Total_Claim_Amount.quantile(0.25)
LLT=df.Total_Claim_Amount.quantile(0.25)-iqrT*1.5
ULT=df.Total_Claim_Amount.quantile(0.75)+iqrT*1.5

df_Total_Claim_Amount=pd.DataFrame(np.where(df.Total_Claim_Amount>ULT,ULT,np.where(df.Total_Claim_Amount<LLT,LLT,df.Total_Claim_Amount)))
sns.boxplot(df_Total_Claim_Amount)


#############################

print(type(df_Customer_Lifetime_Value))
print(type(df))

df.replace({'Customer_Lifetime_Value':'df_Customer_Lifetime_Value','Months_Since_Last_Claim':'df_Months_Since_Last_Claim','Number_of_Policies':'df_Number_of_Policies','Total_Claim_Amount':'df_Total_Claim_Amount'})

df.columns


#### outliers are removed now how to combine it in single daatframe
















