import numpy as np
import scipy.io as sio
import random
# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from EEGModels import EEGNet
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix
from sklearn.preprocessing import StandardScaler

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)
K.set_image_data_format('channels_last')

#数据处理模式；1/2/others
READ_FLAG=2
#数据路径
raw_fname = '/Users/yeye/Downloads/code/s24_reference.cnt'
X_path = 'data/S24_features_pre_12s.pkl'
y_path = 'data/S24_labels_pre_12s.pkl'
#preprocessed_path = 'data/cleandata_sub24.mat'
#preprocessed_path = 'data'
preprocessed_path = 'enroll_data'

#模型保存路径
#CKP_PATH = './tmp/checkpoint_intents_k5_82_14s.h5'
#CKP_PATH = './tmp/checkpoint_intents_k12_82_14s.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_82_14s.h5'
#CKP_PATH = './tmp/checkpoint_intents_k64_82_14s.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_41_14s.h5'
#CKP_PATH = './tmp/checkpoint_intents_k64_82_15s_batch8.h5'
#CKP_PATH = './save_models/checkpoint_intents_k64_82_15s_batch16.h5'
CKP_PATH = './save_models/checkpoint_intents_k25_81_15s_batch4_norm_2_noexpand.h5'

#subs = ["sub01", "sub02", "sub03", "sub04", "sub06", 'sub07','sub10','sub11','sub12','sub13','sub24']
subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']
##################### Process, filter and epoch the data ######################
# Set parameters and read data
def preprocess(raw_fname):
    # Setup for reading the raw data
    raw = io.read_raw_cnt(raw_fname, preload=True, verbose=False)
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
    '''
    # Read epochs
    #tmin, tmax = -0.2, 1.8   #获取-0.2~1s内的数据
    tmin, tmax = -0., 1.2
    #11（JG）--Blue；12（MM）--Red；13（JY）--Green
    event_id = dict(JG=2, MM=3, JY=4)
    epochs = mne.Epochs(raw_for_ica, events, event_id, tmin, tmax, baseline=None,
                    picks=picks, preload=True, verbose=False)
    labels = epochs.events[:, -1]

    # extract raw data. scale by 1000 due to scaling sensitivity in deep learning
    epochs = epochs.resample(128)
    X = epochs.get_data()*1000 # format is in (trials, channels, samples)
    y = labels
    return X, y

def read_from_mat(file_path):
    raw_data = sio.loadmat(file_path)
    data = raw_data['data_2visual']
    chas_arr = data['label'][0][0]
    fsample_arr = data['fsample'][0][0]
    features_arr = data['trial'][0][0].T
    label_arr = data['trialinfo'][0][0]
    return chas_arr,fsample_arr,features_arr,label_arr

if READ_FLAG == 1:
    with open(X_path , "rb") as fx:
        X = pickle.load(fx)
    with open(y_path , "rb") as fy:
        y = pickle.load(fy)
elif READ_FLAG == 2:
    sub_num = 0
    for sub in subs: 
        temp_features = []
        temp_labels = []
        data_path = preprocessed_path + '/cleandata_' + sub + '.mat' 
        chas_arr,fsample_arr, features_arr,label_arr = read_from_mat(data_path)
        n_channels, fs = chas_arr.shape[0], fsample_arr[0][0]
        assert features_arr.shape[0] == label_arr.shape[0]       
        nSamples = features_arr.shape[0]
        #n_timepoints = int(fs*1.7+1)   #-0.2-1.5s
        #n_timepoints = int(fs*1.4+1)    #-0.2-1.2s
        #n_timepoints = int(fs*1.2+1)    #0-1.2s
        n_timepoints = int(fs*1.5+1)    #0-1.5s
        #n_timepoints = int(fs*0.8+1)    #0.4-1.2s
        #n_timepoints = int(fs*1.1+1)    #0.4-1.5s
        #n_timepoints = int(fs*1.0+1)   #0.5-1.5s
        #n_timepoints = int(fs*0.5+1)   #1.0-1.5s
        print(nSamples,n_timepoints)
        for i in range(nSamples):
            temp_features.append(np.zeros([n_channels, n_timepoints]))
            temp_labels.append(np.zeros([1]))
        event_dic = {'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}

        for j in range(nSamples):
            #temp_features[j][:,:] = features_arr[j][0] #-0.2-1.5s
            #temp_features[j][:,:] = features_arr[j][0][:,:701]  #-0.2-1.2s
            #temp_features[j][:,:] = features_arr[j][0][:,100:701]  #0.-1.2s
            temp_features[j][:,:] = features_arr[j][0][:,100:] #0-1.5s
            #temp_features[j][:,:] = features_arr[j][0][:,300:701]  #0.4-1.2s
            #temp_features[j][:,:] = features_arr[j][0][:,300:] #0.4-1.5s
            #temp_features[j][:,:] = features_arr[j][0][:,600:] #1.0-1.5s
            temp_labels[j] = event_dic[str(label_arr[j][0])]

        if sub_num == 0:
            X = np.array(temp_features)
            #X = np.array(temp_features)*1000
            y = np.array(temp_labels)
        else:
            X = np.concatenate((X,np.array(temp_features)), axis=0)
            y = np.concatenate((y,np.array(temp_labels)), axis=0)
        sub_num += nSamples
        print(sub_num, X.shape,y.shape)

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
else:
    X, y = preprocess(raw_fname)
    with open(X_path, 'wb') as fx:
        pickle.dump(X, fx)
    with open(y_path, 'wb') as fy:
        pickle.dump(y, fy)
  
# take 80/20/20 percent of the data to train/validate/test
split_factor = 0.90
X_train      = X[0:int(X.shape[0]*split_factor),]
X_test       = X[int(X.shape[0]*split_factor):,]
Y_test       = y[int(X.shape[0]*split_factor):]
'''
#归一化
scaler = StandardScaler()
num_instances, num_features, num_time_steps = X_train.shape
X_train = np.reshape(X_train, newshape=(-1, num_features))
#X_train = scaler.fit_transform(X_train)
#X_train = np.reshape(X_train, newshape=(num_instances, num_features, num_time_steps))
scaler.fit_transform(X_train)
num_instances, num_features, num_time_steps = X_test.shape
X_test = np.reshape(X_test, newshape=(-1, num_features))
X_test = scaler.transform(X_test)
X_test = np.reshape(X_test, newshape=(num_instances, num_features, num_time_steps))
'''
# convert labels to one-hot encodings.
Y_test       = np_utils.to_categorical(Y_test-2)
'''
#数据随机化
index_shuf = [i for i in range(X.shape[0])]
random.shuffle(index_shuf)
X = np.array([X[i] for i in index_shuf])
y = np.array([y[i] for i in index_shuf])
'''
#采样率 500
#kernels, chans, samples = 1, 60, 851 #-0.2-1.5s
#kernels, chans, samples = 1, 60, 701 #-0.2-1.2s
#kernels, chans, samples = 1, 60, 601 #0-1.2s
kernels, chans, samples = 1, 60, 751 #0-1.5s
#kernels, chans, samples = 1, 60, 401 #0.4-1.2s
#kernels, chans, samples = 1, 60, 551 #0.4-1.5s
#kernels, chans, samples = 1, 60, 601 #-0.2-1.s

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
print(X_test.shape[0], 'test samples') 

model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 25, F1 = 8, D = 1, F2 = 8, 
               dropoutType = 'Dropout')

# load optimal weights
model.load_weights(CKP_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
names = ['JG', 'MM', 'JY']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
#plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-4,2')
plt.show()
