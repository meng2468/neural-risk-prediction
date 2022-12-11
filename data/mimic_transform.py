# %%
import pandas as pd
import os

file_name = os.path.join('../Data/mimiiciv',os.listdir('../Data/mimiiciv')[0])

# %%
file_name

# %%
df = pd.read_excel(file_name)

# %%

import re

ts_features = set(re.findall('(\w+\_\d+)\s',' '.join(df.columns)+' '))
measurements = set([re.findall('(\w+\_*\w*)\_\d+', x)[0] for x in ts_features])
cat_features = list(df.drop(list(ts_features), axis=1).columns.values)


# %%
df_cat = df[cat_features]


# %%
df_cat['90_days_survival'].value_counts()

# %%
df['stay_id'].nunique()

# %%
df.subject_id.nunique()

# %%
print('Splitting out individual measurements')

df_singles = {}
for x in measurements:
    if x == 'lactate':
        df_x = df[['stay_id'] + [y for y in ts_features if (y.startswith(x+'_')) and (not y.startswith(x+'_1_'))]]
    else:
        df_x = df[['stay_id'] + [y for y in ts_features if y.startswith(x+'_')]]
    df_singles[x] = df_x


# %%
df_singles.keys()

# %%
print('Renaming columns and ts')

df_msingles = {}
for k,v in df_singles.items():
    v = pd.melt(v, id_vars=[v.columns.values[0]], value_vars=list(v.columns.values))
    v['t'] = v.variable.apply(lambda x: re.findall('\_(\d+)', x)[0])   
    if v['t'].nunique() == 15:
        v['variable'] = k
        df_msingles[k] = v
    else:
        print('Not time series but grouped here', k)

# %%
df_reformat = pd.concat([v for k, v in df_msingles.items()], ignore_index=True)
df_reformat = df_reformat.sort_values(by=['stay_id','t', 'variable'])[['stay_id','t','variable','value']]


# %%
patient_to_id = {}
patients = df['stay_id'].drop_duplicates()
i = 0
for x in patients:
    patient_to_id[x] = i
    i += 1


# %%
df_cat['id'] = df_cat['stay_id'].apply(lambda x: patient_to_id[x])
df_cat = df_cat.set_index('id')

df_reformat['id'] = df_reformat['stay_id'].apply(lambda x: patient_to_id[x])
df_reformat = df_reformat.set_index('id')

# %%
print('Saving to csvs')
df_reformat.to_csv('mimic_prepared_X.csv')

# %%
df_cat.to_csv('mimic_prepared_Y.csv')


# %%


# %%



