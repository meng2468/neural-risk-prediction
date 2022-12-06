#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import os

file_name = os.path.join('eicu',os.listdir('eicu')[0])


# In[3]:


df = pd.read_excel(file_name)


# In[120]:


import re

ts_features = set(re.findall('(\w+\_\d+)\s',' '.join(df.columns)+' '))
measurements = set([re.findall('(\w+\_*\w*)\_\d+', x)[0] for x in ts_features])
cat_features = list(df.drop(list(ts_features), axis=1).columns.values) + ['sxyl_lactate_1', 'sxyl_lactate_2', 'labresultoffset_1', 'labresultoffset_2']


# In[123]:


df_cat = df[cat_features]


# In[125]:


df_cat['survival_90days'].describe()


# In[159]:


df_singles = {}
for x in measurements:
    df_x = df[['patientunitstayid'] + [y for y in ts_features if y.startswith(x)]]
    df_singles[x] = df_x


# In[160]:


df_msingles = {}
for k,v in df_singles.items():
    v = pd.melt(v, id_vars=[v.columns.values[0]], value_vars=list(v.columns.values))
    v['t'] = v.variable.apply(lambda x: re.findall('\_(\d+)', x)[0])   
    if v['t'].nunique() == 15:
        v['variable'] = k
        df_msingles[k] = v
    else:
        print('Not time series but grouped here', k)

    


# In[161]:


df_reformat = pd.concat([v for k, v in df_msingles.items()], ignore_index=True)
df_reformat = df_reformat.sort_values(by=['patientunitstayid','t', 'variable'])[['patientunitstayid','t','variable','value']]

df_reformat


# In[162]:


import numpy as np

patient_to_id = {}
patients = df['patientunitstayid'].drop_duplicates()
i = 0
for x in patients:
    patient_to_id[x] = i
    i += 1


# In[163]:


df_cat['id'] = df_cat['patientunitstayid'].apply(lambda x: patient_to_id[x])
df_cat = df_cat.set_index('id')
df_reformat['id'] = df_reformat['patientunitstayid'].apply(lambda x: patient_to_id[x])
df_reformat = df_reformat.set_index('id')


# In[164]:


df_reformat.to_csv('eicu_prepared_X.csv')


# In[165]:


df_cat.to_csv('eicu_prepared_Y.csv')


# In[ ]:




