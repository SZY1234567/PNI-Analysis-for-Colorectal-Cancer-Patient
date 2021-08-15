# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 21:00:17 2021

@author: Zhuoyan Shen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter
from lifelines.utils import median_survival_times
from lifelines.statistics import logrank_test
from lifelines.datasets import load_waltons


data = pd.read_csv('data.csv')
AD = data.AD.astype('int')
data['AD'] = AD
data.set_index('AD',inplace=True,drop='True')

print(data.columns)

df = data.dropna(subset=['survival1','three_year_survival1'])

X = df['survival1']/30
y = df['three_year_survival1']
model = KaplanMeierFitter()

model.fit(durations=X, event_observed=y) 
model.plot()
plt.title('KM Curve')
plt.show()

def plot_KM(data,label,duration,event):
    
    data.dropna(subset=[duration,event],inplace=True)
    df1 = data[data[label]==0]
    df2 = data[data[label]==1]
    X1 = df1[duration]/30
    X2 = df2[duration]/30
    y1 = df1[event]
    y2 = df2[event]
    
    ax = plt.subplot(111)
    
    model.fit(X1, y1, label='{} < THRESHOLD'.format(label))
    model.plot(ax=ax,show_censors=True, censor_styles={'ms': 6, 'marker': 's'})
    model.fit(X2, y2, label='{} >= THRESHOLD'.format(label))
    model.plot(ax=ax,show_censors=True, censor_styles={'ms': 6, 'marker': 's'})
    
    test = logrank_test(X1, X2, y1, y2,weightings='wilcoxon')
    test.print_summary()
    test = logrank_test(X1, X2, y1, y2,weightings='tarone-ware')
    test.print_summary()
    test = logrank_test(X1, X2, y1, y2,weightings='peto')
    test.print_summary()
    test = logrank_test(X1, X2, y1, y2,weightings='fleming-harrington',p=0,q=0)
    test.print_summary()
    
    median_confidence_interval = median_survival_times(model.confidence_interval_)
    print('Suvival months with 95% confidence intervial: {}'.format(median_confidence_interval))



plot_KM(data=df,label='PNI_stat',duration='survival1',event='three_year_survival1')
plt.title('KM curves in terms of high or low PNI (Start: Date of Diagnosis)')
plt.show()
#plot_KM(data=df,label='PNI_stat',duration='survival2',event='three_year_survival2')
#plt.title('KM curves in terms of high or low PNI (Start: Date of Test)')
#plt.show()
#%%
plot_KM(data=df,label='PNI_stat',duration='survival1',event='five_year_survival1')
plt.title('KM curves in terms of high or low PNI (Start: Date of Diagnosis)')
plt.show()
#plot_KM(data=df,label='PNI_stat',duration='survival2',event='five_year_survival2')
#plt.title('KM curves in terms of high or low PNI (Start: Date of Test)')
#plt.show()
    
    
