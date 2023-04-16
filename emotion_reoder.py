from sklearn import metrics
from sklearn import preprocessing
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import h5py
import scipy.stats
from scipy.stats import pearsonr
from neurora.stuff import limtozero


subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
nsubs = len(subs)

#LOAD DATASET
actList = ['JG', 'MM', 'JY']
#actList = ['MM', 'JG']
# load data
#search_list = ['duration','f2meanfrequency', 'f3meanfrequency','meanintensity','pitchMax', 'pitchrange']
#search_list = ['meanintensity','pitchMax','duration','f3meanfrequency']
#search_list = ['speaker', 'item', 'speechact', 'attitude','valence','meanintensity',
#'duration','f2meanfrequency', 'f3meanfrequency','pitchMax', 'pitchrange']

df_bhv = pd.read_excel('./index_order.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
#df_index = pd.read_excel('./index_order.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头
#df_index = df_index.apply(lambda x:12*np.log2(x/100) if x.name in ['pitchMax','pitchMin', 'pitchrange','f1meanfrequency', 'f2meanfrequency', 'f3meanfrequency'] else x)
df_bhv.sort_values(by="attitude" , inplace=True, ascending=True) 
#dataframe = df_bhv.loc[:,search_list]
count = 1
df_bhv['trial_ID_emotion'] = 0
for index, row in df_bhv.iterrows():
    trial_id = row['trial_ID'].split('_')
    print(count)
    #intent_class = trial_id.split('_')[-1]
    if count <= 48:
        df_bhv.loc[index, 'trial_ID_emotion'] = ('_').join(trial_id[:-1])+'_JG'
    elif 48 < count <= 96:
        df_bhv.loc[index, 'trial_ID_emotion'] = ('_').join(trial_id[:-1])+'_MM'
    elif count > 96:
        df_bhv.loc[index, 'trial_ID_emotion'] = ('_').join(trial_id[:-1])+'_JY'
    count += 1
    #print(index, trial_id, intent_class)
dataframe = df_bhv
dataframe.to_excel("emotion_reorder.xlsx",na_rep=False)
