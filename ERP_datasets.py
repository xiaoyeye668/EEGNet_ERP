import numpy as np
import random
# mne imports
import mne
from mne import io
from sklearn.model_selection import train_test_split

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

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)

#preprocessed_path = 'data/cleandata_sub24.mat'
#preprocessed_path = 'data'
preprocessed_path = 'enroll_data'

READ_FLAG=1
X_path = 'data/S24_features_pre_12s.pkl'
y_path = 'data/S24_labels_pre_12s.pkl'
#subs = ["sub01"]
#subs = ["sub01", "sub02", "sub03", "sub04", "sub06", 'sub07','sub10','sub11','sub12','sub13','sub24']

sub_lists = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']

##################### Process, filter and epoch the data ######################
# Set parameters and read data
def preprocess(raw_fname):
    # Setup for reading the raw data
    raw = io.read_raw_cnt(raw_fname, preload=True, verbose=False)
    #io.read_raw_fif(raw_fname)
    print(raw.info)

    #高低通滤波
    raw = raw.filter(l_freq=0.1, h_freq=40)
    #默认method='fir'，使用IIR则修改为'iir'
    #raw = raw.filter(l_freq=0.1, h_freq=30, method='iir')
    #raw.plot_psd()

    #首先，需要确定分段需要用到的markers, 查看数据中的markers
    print(raw.annotations)
    #事件信息数据类型转换
    # 将Annotations类型的事件信息转为Events类型
    events, event_id = mne.events_from_annotations(raw)
    #events为记录时间相关的矩阵，event_id为不同markers对应整型的字典信息
    print(events.shape, event_id)
    #{'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}

    #raw.info['bads'] = ['MEG 2443']  # set bad channels
    #raw.info['bads'].extend(['M1', 'M2', 'CB1', 'CB2','REF']) 
    raw.info['bads'].extend(['FC1', 'HEO', 'REF','VEO'])
    picks = mne.pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                       exclude='bads')

    '''
    独立成分分析（ICA）
    '''
    #ica = mne.preprocessing.ICA(n_components=22, random_state=97, max_iter=800)
    ica = mne.preprocessing.ICA(max_iter=50)
    raw_for_ica = raw.copy()
    ica.fit(raw_for_ica)
    #ica.plot_sources(raw_for_ica)
    # 设定要剔除的成分序号
    ica.exclude = [55,56,57,58,59,60,61,62]   # details on how we picked these are omitted here
    # 应用到脑电数据上
    ica.apply(raw_for_ica)
    #raw.plot()

    '''
    数据分段
    基于Events对数据进行分段
    这里提取刺激前1秒到刺激后1秒的数据
    取baseline时间区间为刺激前0.5s到刺激出现
    并进行卡阈值，即在epoch中出现最大幅值与最小幅值的差大于2×10^-4则该epoch被剔除
    注意：这里的阈值设置较大，一般数据质量佳的情况下推荐设置为5×10^-5到1×10^4之间
    epochs = mne.Epochs(raw, events, event_id, tmin=-1, tmax=2, baseline=(-0.5, 0), 
                    preload=True, reject=dict(eeg=2e-4))
    '''
    # Read epochs
    #tmin, tmax = -0.2, 1.5   #获取-0.2~1.5s内的数据
    tmin, tmax = -0., 1.2
    #11（JG）--Blue；12（MM）--Red；13（JY）--Green
    event_id = dict(JG=2, MM=3, JY=4)
    epochs = mne.Epochs(raw_for_ica, events, event_id, tmin, tmax, baseline=None,
                    picks=picks, preload=True, verbose=False)
    labels = epochs.events[:, -1]

    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    epochs = epochs.resample(128)   #sf=128,最高可分析到频率64
    X = epochs.get_data()
    #psd,freqs = mne.time_frequency.psd_multitaper()
    #psd = 10 * np.log10(np.mean(np.mean(psd, axis=0), axis=0) * 1e12)
    #X = epochs.get_data()*1000 # format is in (trials, channels, samples)
    y = labels
    print(X.shape,y.shape)
    return X, y

def read_from_mat(file_path):
    raw_data = sio.loadmat(file_path)
    data = raw_data['data_2visual']
    chas_arr = data['label'][0][0]
    fsample_arr = data['fsample'][0][0]
    features_arr = data['trial'][0][0].T    #features_arr shape:[n_trials, 1]  
    label_arr = data['trialinfo'][0][0]
    return chas_arr,fsample_arr,features_arr,label_arr

fs = 500
tmin, tmax = -0.2, 1.5 #-0.2-1.5s
n_subjects = 28
n_maxtrials = 144
n_channels = 60
n_timepoints = int(fs*(tmax-tmin)+1)   
#X_tmp = np.zeros([n_subjects, n_maxtrials, n_channels, n_timepoints])
#Y_tmp = np.zeros([n_subjects, n_maxtrials])

if READ_FLAG == 1:
    sub_num = 0
    for ii in range(n_subjects):
        sub =  sub_lists[ii]
        temp_data = []
        temp_labels = []
        data_path = preprocessed_path + '/cleandata_' + sub + '.mat' 
        chas_arr,fsample_arr, features_arr, label_arr = read_from_mat(data_path)
        n_channels, fs = chas_arr.shape[0], fsample_arr[0][0]
        assert features_arr.shape[0] == label_arr.shape[0]       
        nSamples = features_arr.shape[0]
        for i in range(nSamples):
            temp_data.append(np.zeros([n_channels, n_timepoints]))
            temp_labels.append(np.zeros([1]))
        event_dic = {'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}
               
        for j in range(nSamples):
            temp_data[j][:,:] = features_arr[j][0] #-0.2-1.5s
            temp_labels[j] = event_dic[str(label_arr[j][0])]


        if sub_num == 0:
            #X = np.array(temp_data)
            #scale by 1000 due to scaling sensitivity in deep learning
            X = np.array(temp_data)*1000
            y = np.array(temp_labels)
        else:
            #X = np.concatenate((X,np.array(temp_data)), axis=0)
            X = np.concatenate((X,np.array(temp_data)*1000), axis=0)
            y = np.concatenate((y,np.array(temp_labels)), axis=0)
        sub_num += nSamples
        print(sub_num, X.shape,y.shape)
        
else:
    X, y = preprocess(raw_fname)
    with open(X_path, 'wb') as fx:
        pickle.dump(X, fx)
    with open(y_path, 'wb') as fy:
        pickle.dump(y, fy)

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size = 0.9, random_state=8, stratify=y)
X_validate, y_validate = X_test, y_test

# convert labels to one-hot encodings.
Y_train      = np_utils.to_categorical(y_train-2)
Y_validate   = np_utils.to_categorical(y_validate-2)
Y_test       = np_utils.to_categorical(y_test-2)

feature_save_path = './datasets_cross_new/datasets_cross_17s_split90/'

np.save(feature_save_path+'X_train.npy', X_train)
np.save(feature_save_path+'Y_train.npy', Y_train)
print("save {}.npy done".format(X_train.shape))
np.save(feature_save_path+'X_validate.npy', X_validate)
np.save(feature_save_path+'Y_validate.npy', Y_validate)
print("save {}.npy done".format(X_validate.shape))
np.save(feature_save_path+'X_test.npy', X_test)
np.save(feature_save_path+'Y_test.npy', Y_test)
print("save {}.npy done".format(X_test.shape))
