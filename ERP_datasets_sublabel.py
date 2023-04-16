import numpy as np
import random
import pandas as pd
# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from EEGModels import EEGNet
import tensorflow as tf
#tf.compat.v1.disable_eager_execution()
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras import optimizers

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix
import pickle
import scipy.io as sio
import h5py
from collections import defaultdict

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)

#preprocessed_path = 'data/cleandata_sub24.mat'
#preprocessed_path = 'data'
preprocessed_path = 'enroll_data'

READ_FLAG=1

#subs = ["sub01"]
#subs = ["sub01", "sub02", "sub03", "sub04", "sub06", 'sub07','sub10','sub11','sub12','sub13','sub24']

train_subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
df_voice = pd.read_excel('./behave_28subs_new.xlsx', sheet_name=0, header=0)   #读取第一张表，取第一行作表头

def read_from_mat(file_path):
    raw_data = sio.loadmat(file_path)
    data = raw_data['data_2visual']
    chas_arr = data['label'][0][0]
    fsample_arr = data['fsample'][0][0]
    features_arr = data['trial'][0][0].T
    label_arr = data['trialinfo'][0][0]
    return chas_arr,fsample_arr,features_arr,label_arr

if READ_FLAG == 1:
    sub_num = 0
    dictmp = defaultdict()
    for sub in train_subs: 
        subid = int(sub[3:])
        df_data = df_voice[df_voice['subject']==subid]
        temp_features = []
        temp_labels = []
        data_path = preprocessed_path + '/cleandata_' + sub + '.mat' 
        chas_arr,fsample_arr, features_arr,label_arr = read_from_mat(data_path)
        n_channels, fs = chas_arr.shape[0], fsample_arr[0][0]
        assert features_arr.shape[0] == df_data.shape[0]
        nSamples = features_arr.shape[0]
        n_timepoints = int(fs*1.7+1)   #-0.2-1.5s
        print(nSamples, n_timepoints)
        for i in range(nSamples):
            temp_features.append(np.zeros([n_channels, n_timepoints]))
            temp_labels.append(np.zeros([1]))
        #event_dic = {'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}

        for j in range(nSamples):
            temp_features[j][:,:] = features_arr[j][0] #-0.2-1.5s
            sub_label = df_data.iloc[j]['behavior intention']
            if -4<= sub_label <=-2:
                temp_labels[j] = 2
                if 'JG' not in dictmp.keys():
                    dictmp['JG'] =1
                else:
                    dictmp['JG'] += 1
            elif -1<= sub_label <=1:
                temp_labels[j] = 3
                if 'MM' not in dictmp.keys():
                    dictmp['MM'] =1
                else:
                    dictmp['MM'] += 1
            elif 2<= sub_label <=4:
                temp_labels[j] = 4
                if 'JY' not in dictmp.keys():
                    dictmp['JY'] =1
                else:
                    dictmp['JY'] += 1
                 
        if sub_num == 0:
            #X = np.array(temp_features)
            X = np.array(temp_features)*1000
            y = np.array(temp_labels)
        else:
            #X = np.concatenate((X,np.array(temp_features)), axis=0)
            X = np.concatenate((X,np.array(temp_features)*1000), axis=0)
            y = np.concatenate((y,np.array(temp_labels)), axis=0)
        sub_num += nSamples
        print(sub_num, X.shape,y.shape)
    print(dictmp)
    '''
    #Epochs object from numpy array.
    data = X
    ch_names = ch_names = [str(i) for i in range(1,61)] #通道名称
    sfreq = 500 #采样率
    info = mne.create_info(ch_names, sfreq, ch_types = "eeg") #创建信号的信息
    #event_dic = {'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}
    #events.shape (n_events, 3) [timestamp,0,label]
    events = np.zeros([y.shape[0],3])
    events[:,-1] = y
    events[:,0] = np.array([int(i) for i in range(y.shape[0])])
    events=events.astype(int)
    event_id = dict(JG=2, MM=3, JY=4)
    epochs = mne.EpochsArray(data, info, events=events, tmin=0, event_id=event_id)
    #labels = epochs.events[:, -1]
    #epochs = epochs.resample(128)
    #X = epochs.get_data() # format is in (trials, channels, samples)
    X = epochs.get_data()*1000
    y = epochs.events[:, -1]
    print('<<<<<<<<<<<<')
    print(X.shape)
    '''
#数据随机化
index_shuf = [i for i in range(X.shape[0])]
random.Random(0).shuffle(index_shuf)
X = np.array([X[i] for i in index_shuf])
y = np.array([y[i] for i in index_shuf])

# take 80/20/20 percent of the data to train/validate/test
split_factor = 0.90
X_train      = X[0:int(X.shape[0]*split_factor),]
Y_train      = y[0:int(X.shape[0]*split_factor)]
#X_train      = X
#Y_train      = y
X_validate_1   = X[int(X.shape[0]*split_factor):,]
Y_validate_1   = y[int(X.shape[0]*split_factor):]
X_validate_2   = X_train[int(X_train.shape[0]*0.95):,]
Y_validate_2   = Y_train[int(X_train.shape[0]*0.95):]
X_validate = np.concatenate((X_validate_1,X_validate_2))
Y_validate = np.concatenate((Y_validate_1, Y_validate_2))
X_test       = X_validate
Y_test       = Y_validate
'''
#数据随机化
index_shuf = [i for i in range(X_train.shape[0])]
random.shuffle(index_shuf)
X_train = np.array([X_train[i] for i in index_shuf])
Y_train = np.array([Y_train[i] for i in index_shuf])
'''
# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(Y_train-2)
Y_validate   = np_utils.to_categorical(Y_validate-2)
Y_test       = np_utils.to_categorical(Y_test-2)

feature_save_path = './datasets_cross_sublabel/datasets_cross_17s/'

np.save(feature_save_path+'X_train.npy', X_train)
np.save(feature_save_path+'Y_train.npy', Y_train)
print("save {} X_train done".format(X_train.shape))
np.save(feature_save_path+'X_validate.npy', X_validate)
np.save(feature_save_path+'Y_validate.npy', Y_validate)
print("save {} X_valid done".format(X_validate.shape))
np.save(feature_save_path+'X_test.npy', X_test)
np.save(feature_save_path+'Y_test.npy', Y_test)
print("save {} X_test done".format(X_test.shape))
