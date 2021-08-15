# -*- coding: utf-8 -*-
"""
Created on Fri Feb 5 09:28:59 2021

@author: zhuoy
"""

from lifelines import CoxPHFitter
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data.csv')
AD = data.AD.astype('int')
data['AD'] = AD
data.set_index('AD',inplace=True,drop='True')

print(data.columns)

categorical_variables = [
    'gender','blood_group','second_cancer', 'MLH1', 'MSH2', 'MSH6', 'PMS2', 
    'MMR1','MMR2', 'MSI', 'RAS', 'BRAF','cancer_type','differentiation', 
    'pathological_type','PNI', 'VI','T_stage', 'N_stage', 'M_stage', 'AJCC_stage',
    #'liver_m', 'lung_m', 'peritoneum_m', 'abdomen_m', 'pelvic_m', 'bone_m' ,
    #'brain_m', 'other_m'
    ]

numerical_variables = [
    'age_at_diagnosis',
    #'serum albumin level (g/L)', 'total lymphocyte count (/L)', 
    #'total neutrophil count(/L)','total monocyte count(/L)',
    #'total platelet count(/L)',
    'height', 'weight',
    #'BMI', 'BSA',
    'tumour_size',
    #'num_of_lymph_nodes', 'num_of_positive_lymph_nodes',
    'LMR', 'NLR', 'PLR', 'PNI_'
    ]


def create_subset(data,duration_label,event_label):

    df = data[
        categorical_variables+
        numerical_variables
        ]
 
    df['duration'] = data[duration_label]/30
    df['event'] = data[event_label]
    
    df.dropna(axis='columns', thresh=30,inplace=True)
    df.fillna(method='bfill',inplace=True)
    df.fillna(method='ffill',inplace=True)
    
    return df
    
    
def Cox_Regression(data):
    
    model = CoxPHFitter(penalizer=0.1)
    model.fit(data,'duration', 'event',show_progress=True)
    model.print_summary()
    

Cox_Regression(create_subset(data,'survival1','three_year_survival1'))
Cox_Regression(create_subset(data,'survival1','five_year_survival1'))










