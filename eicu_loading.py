import pandas as pd
import os

from sklearn import preprocessing
import numpy as np
import re


if __name__ == '__main__':
    file_name = os.path.join('../Data/eicu',os.listdir('../Data/eicu')[0])

    print('Pulling data from', file_name)
    df = pd.read_excel(file_name)
    df = df.query('patientunitstayid != 708912')

    ts_features = set(re.findall('(\w+\_\d+)\s',' '.join(df.columns)+' '))
    measurements = set([re.findall('(\w+\_*\w*)\_\d+', x)[0] for x in ts_features])
    cat_features = list(df.drop(list(ts_features), axis=1).columns.values) + ['sxyl_lactate_1', 'sxyl_lactate_2', 'labresultoffset_1', 'labresultoffset_2']

    df_cat = df[cat_features]

    df_singles = {}
    for x in measurements:
        df_x = df[['patientunitstayid'] + [y for y in ts_features if y.startswith(x)]]
        df_singles[x] = df_x

    df_msingles = {}
    for k,v in df_singles.items():
        v = pd.melt(v, id_vars=[v.columns.values[0]], value_vars=list(v.columns.values))
        v['t'] = v.variable.apply(lambda x: re.findall('\_(\d+)', x)[0]).astype(int) 
        if v['t'].nunique() == 15:
            v['variable'] = k
            df_msingles[k] = v
        else:
            print('Not time series but grouped here', k)

    df_t_pheno =  pd.concat([v for k, v in df_msingles.items()], ignore_index=True)

    intervention = ['treatment',
        'treatmentoffset', 'mbp', 'mbp_offset', 'crystalloids',
        'crystalloidsoffset', 'vasoactive_drugs', 'vasoactive_drugs_offset',
        'invasive_line', 'invasive_lineoffset', 'hydrocortisone',
        'hydrocortisoneoffset', 'nursecare', 'nursecareoffset', 'dialysis',
        'dialysisoffset', 'survival_90days', 'sxyl_lactate_1', 'sxyl_lactate_2',
        'labresultoffset_1', 'labresultoffset_2']

    ts_features = list(set(re.findall('(\w+\_\d+)\s',' '.join(df.columns)+' ')))
    df_cat = df.drop(columns=list(ts_features)+intervention+['microlab','chartime'])

    enc = preprocessing.OrdinalEncoder()
    enc.fit(df_cat[['gender','ethnicity','age']])
    df_cat[['gender','ethnicity','age']] = enc.transform(df_cat[['gender','ethnicity','age']])

    times = df_t_pheno.t.unique()
    df_cat_pivot = pd.melt(df_cat, id_vars='patientunitstayid')
    df_var_t = pd.melt(pd.DataFrame({x: times for x in df_cat_pivot.variable.unique()})).rename(columns={'value':'t'})
    df_cat_pheno = pd.merge(df_var_t, df_cat_pivot, left_on='variable', right_on='variable').sort_values(by=['patientunitstayid','t','variable'])

    df_final_x = pd.concat([df_t_pheno, df_cat_pheno]).sort_values(by=['patientunitstayid','t','variable'])
    df_final_y = df[['patientunitstayid', 'survival_90days']].replace({'alive':0, 'death':1})

    df_final_x.to_csv('eicu_prepared_x.csv', index=False)
    df_final_y.to_csv('eicu_prepared_y.csv', index=False)
