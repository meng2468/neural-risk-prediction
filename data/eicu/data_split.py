import pandas as pd
from sklearn.model_selection import train_test_split

data_x = pd.read_csv('data/eicu/eicu_prepared_x.csv',index_col='id')
labels = pd.read_csv('data/eicu/eicu_prepared_y.csv', index_col='id')

train, test = train_test_split(data_x.index.unique(), train_size=.75, stratify=labels.survival_90days)
test, val = train_test_split(test, train_size=.6, stratify=labels.loc[test].survival_90days)

def reset_ids(df):
    patient_to_id = {}
    patients = df['patientunitstayid'].drop_duplicates()
    i = 0
    for x in patients:
        patient_to_id[x] = i
        i += 1
    df = df.reset_index()
    df['id'] = df['patientunitstayid'].apply(lambda x: patient_to_id[x])
    df = df.set_index('id')
    return df

reset_ids(data_x.loc[train]).to_csv('data/eicu/eicu_train_x.csv')
reset_ids(data_x.loc[test]).to_csv('data/eicu/eicu_test_x.csv')
reset_ids(data_x.loc[val]).to_csv('data/eicu/eicu_val_x.csv')

reset_ids(labels.loc[train]).to_csv('data/eicu/eicu_train_y.csv')
reset_ids(labels.loc[test]).to_csv('data/eicu/eicu_test_y.csv')
reset_ids(labels.loc[val]).to_csv('data/eicu/eicu_val_y.csv')