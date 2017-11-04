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
corr = studentDF.corr()

for col in corr:
    x=corr.drop(col)
    print(col + " has the strongest positive correlation to  " + x[col].idxmax())
    print(col + " has the strongest negative correlation to  " + x[col].idxmin())
    print()
sns.heatmap(corr)