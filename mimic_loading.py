import pandas as pd
import os

import re
from sklearn import preprocessing

if __name__ == '__main__':
    file_name = os.path.join('../Data/mimiiciv',os.listdir('../Data/mimiiciv')[0])

    df = pd.read_excel(file_name)

    # Extract and format time series data

    # Select all columns that have format term_number
    ts_features = list(set(re.findall('(\w+\_\d+)\s',' '.join(df.columns)+' ')))
    # Remove numbers to just get measure names
    measures = list(set(re.findall('(\w+)\_', ' '.join(ts_features))))
    measures = [x for x in measures if 'charttime' not in x]

    # Split dataframe into multiple {measurement: measurement dataframe} format
    df_measures = {}
    for x in measures:
        data = []
        for y in ts_features:
            if re.search(x+'\_\d+',y) != None and re.search(x+'\_\d+',y)[0] == y:
                data.append(y)
        df_measures[x] = df[['stay_id']+data]
    
    # Melt dataframe into long form data
    df_msingles = {}
    for k,v in df_measures.items():
        v = pd.melt(v, id_vars=[v.columns.values[0]], value_vars=list(v.columns.values))
        v['t'] = v.variable.apply(lambda x: int(re.findall(k+'\_(\d+)', x)[0]))   
        v['variable'] = k
        df_msingles[k] = v
    
    # Concatenate all ts data together
    df_t_pheno = pd.concat([v for k, v in df_msingles.items()], ignore_index=True)

    # Extract and format categorical data

    # Get time intervals to duplicate static phenotypes
    times = df_t_pheno.t.unique()

    # Column names containing intervention and ts data
    intervention = ['microbiologyevents', 'microbiologyevents_chartime', 'antibiotic',
       'antibiotic_starttime', 'crystalloids', 'crystalloids_time', 'map_time',
       'pharmacy', 'pharmacy_starttime', 'invasive_line',
       'invasive_line_starttime', 'hydrocortisone', 'hydrocortisone_starttime',
       'ventilation', 'ventilation_starttime', 'rrt', 'rrt_starttime']
    
    ts_features = list(set(re.findall('(\w+\_\d+)\s',' '.join(df.columns)+' ')))

    # Select only categorical and add time taken between arrival and departure
    df_cat = df.drop(columns=list(ts_features)+intervention)
    df_cat['admit_out'] = (df_cat['outtime'] - df_cat['admittime']).dt.total_seconds()/86400
    df_cat['in_out'] = (df_cat['outtime'] - df_cat['intime']).dt.total_seconds()/86400
    df_cat = df_cat.drop(columns=['onsettime', 'admittime','intime','outtime', 'hadm_id','subject_id','90_days_survival'])

    # Reformat non-numerical data
    nn_columns = ['gender','ethnicity']

    enc = preprocessing.OrdinalEncoder().fit(df[nn_columns])
    df_cat[nn_columns] = enc.transform(df[nn_columns])

    # Pivot table and merge with all times to duplicate values
    df_cat_pivot = pd.melt(df_cat, id_vars='stay_id')
    df_var_t = pd.melt(pd.DataFrame({x: times for x in df_cat_pivot.variable.unique()})).rename(columns={'value':'t'})
    
    df_cat_pheno = pd.merge(df_var_t, df_cat_pivot, left_on='variable', right_on='variable').sort_values(by=['stay_id','t','variable'])

    # Concatenate both categorical and ts data for final dataset
    df_final_x = pd.concat([df_t_pheno, df_cat_pheno]).sort_values(by=['stay_id','t','variable'])

    # Take patient survival and stay id and store to final labels
    df_final_y = df[['stay_id', '90_days_survival']].replace({'alive':0, 'death':1})

    df_final_x.to_csv('mimic_prepared_x.csv', index=False)
    df_final_y.to_csv('mimic_prepared_y.csv', index=False)

