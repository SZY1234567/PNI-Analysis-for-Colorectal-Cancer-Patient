# -*- coding: utf-8 -*-
"""
Created on Sat Feb 6 11:31:59 2021

@author: zhuoy
"""


import pandas as pd
import numpy as np
from datetime import datetime 
from sklearn import metrics
import matplotlib.pyplot as plt

data1 = pd.read_csv('stage4.csv').dropna()
data2 = pd.read_csv('stage1_3.csv').dropna()

df = pd.concat( [data1,data2], axis=0 )

patient_list = np.concatenate((data1.AD.to_list(),data2.AD.to_list()))


data = pd.read_csv('pfs.csv')
data.drop(['pfs0', 'pfs1', 'pfs2', 'pfs3', 
           'pfs4','date_of_death','initial_symptom',
           'diabetes', 'high blood pressure', 'hepatitis_B',
           'anemia'], axis=1,inplace=True)

'''
data = data.loc[data['AD'].isin(patient_list)]
df = pd.merge(data, data1, how='left', on='user_id')

'''

df2 = pd.merge(df, data, how='left', on='AD')
df2['LMR']=df2['total lymphocyte count (/L)']/df2['total monocyte count(/L)']
df2['NLR']=df2['total neutrophil count(/L)']/df2['total lymphocyte count (/L)']
df2['PLR']=df2['total platelet count(/L)']/df2['total lymphocyte count (/L)']
df2['PNI_'] = df2['total lymphocyte count (/L)'].apply(lambda x: x*5) + df2['serum albumin level (g/L)']

print(df2.columns)


df2['date_of_test'] = pd.to_datetime(df2['date_of_test'])
df2['date_of_birth'] = pd.to_datetime(df2['date_of_birth'])
df2['date_of_diagnosis'] = pd.to_datetime(df2['date_of_diagnosis'])
df2['date_of_metastasis'] = pd.to_datetime(df2['date_of_metastasis'])
df2.replace('alive',datetime.now(),inplace=True)
df2['DOD'] = pd.to_datetime(df2['DOD'])

df2['survival1'] = df2['DOD'] - df2['date_of_diagnosis']
df2['survival2'] = df2['DOD'] - df2['date_of_test']
df2['survival1'] = df2['survival1'].dt.days
df2['survival2'] = df2['survival2'].dt.days

df2['three_year_survival1'] = df2['survival1']>365*3
df2['three_year_survival2'] = df2['survival2']>365*3
df2['five_year_survival1'] = df2['survival1']>365*5
df2['five_year_survival2'] = df2['survival2']>365*5

def Find_Optimal_Cutoff(TPR, FPR, threshold):
    y = TPR - FPR
    Youden_index = np.argmax(y) 
    optimal_threshold = threshold[Youden_index]
    point = [FPR[Youden_index], TPR[Youden_index]]
    return optimal_threshold, point

def ROC(label, y_prob):
    """
    Receiver_Operating_Characteristic, ROC
    :param label: (n, )
    :param y_prob: (n, )
    :return: fpr, tpr, roc_auc, optimal_th, optimal_point
    """
    fpr, tpr, thresholds = metrics.roc_curve(label, y_prob)
    roc_auc = metrics.auc(fpr, tpr)
    optimal_th, optimal_point = Find_Optimal_Cutoff(TPR=tpr, FPR=fpr, threshold=thresholds)
    return fpr, tpr, roc_auc, optimal_th, optimal_point

fpr, tpr, roc_auc, optimal_th, optimal_point = ROC(df2['three_year_survival1'], df2['PNI_'])

plt.figure(1)
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--")
plt.plot(optimal_point[0], optimal_point[1], marker='o', color='r')
plt.text(optimal_point[0], optimal_point[1], f'Threshold:{optimal_th:.2f}')
plt.title("ROC-AUC of PNI (three year survival)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.show()

df2['PNI_stat'] = df2['PNI_']>optimal_th
df2.to_csv('data.csv',index=False)