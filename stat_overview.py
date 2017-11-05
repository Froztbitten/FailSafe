# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:54:43 2017

@author: Jacob
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

studentDF = pd.read_csv("starfish_data.csv")

# Remove the zipcode entry because it is meaningless for the analysis.
#studentDF.drop('admitted_zip_code', axis=1, inplace=True)
studentDF.drop('person_party_id', axis=1, inplace=True)
studentDF.drop('cms_user_ext_id', axis=1, inplace=True)
studentDF.drop('start_dt', axis=1, inplace=True)
studentDF.drop('end_dt', axis=1, inplace=True)
studentDF.drop('next_start_dt', axis=1, inplace=True)
studentDF.drop('program_start_date', axis=1, inplace=True)
studentDF.drop('program_end_date', axis=1, inplace=True)

studentDF = pd.get_dummies(studentDF)

corr = studentDF.corr()

y = corr['outcome']
for i in corr['outcome']:
    if i > .45 or i < -.45:
        print(i)
        
        

print()
y = corr['prediction']
for i in corr['prediction']:
    if i > .45 or i < -.45:
        print(i)
        
xlist = list()
ylist = list()

'''
for col in corr:
    x=corr.drop(col)
    print(col + " has the strongest positive correlation to  " + x[col].idxmax())7
    print(col + " has the strongest negative correlation to  " + x[col].idxmin())
    print()
    
sns.heatmap(corr)
'''
