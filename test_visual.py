import numpy as np
import scipy.io as sio
# mne imports
import mne
from mne import io
from mne.datasets import sample

# EEGNet-specific imports
from EEGModels import EEGNet
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
#from tensorflow.keras import backend as K
import tensorflow.compat.v1.keras.backend as K
from tensorflow.keras.models import Model

# tools for plotting confusion matrices
from matplotlib import pyplot as plt
from pyriemann.utils.viz import plot_confusion_matrix
from deepexplain.tf.v2_x import DeepExplain

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
preprocessed_path = 'data/cleandata_sub24.mat'

#模型保存路径
#CKP_PATH = './tmp/checkpoint_intents_s24_82_0.5s_tail.h5'
#CKP_PATH = './tmp/checkpoint_intents_s24_82_17s_k5.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_41_14s.h5 '
CKP_PATH = './save_models/checkpoint_intents_k64_82_15s_batch16_nodrop.h5'

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
    temp_features = []   
    chas_arr,fsample_arr, features_arr,label_arr = read_from_mat(preprocessed_path)
    n_channels, fs = chas_arr.shape[0], fsample_arr[0][0]
    assert features_arr.shape[0] == label_arr.shape[0]
    nSamples = features_arr.shape[0]
    #n_timepoints = int(fs*1.7+1)   #-0.2-1.5s
    n_timepoints = int(fs*1.5+1)    #-0.-1.5s
    #n_timepoints = int(fs*1.0+1)    #-0.-1.0s
    #n_timepoints = int(fs*1.2+1)    #-0.2-1.0s
    #n_timepoints = int(fs*1.0+1)   #0.5-1.5s
    #n_timepoints = int(fs*0.5+1)   #1.0-1.5s
    print(nSamples,n_timepoints)
    for i in range(nSamples):
        temp_features.append(np.zeros([n_channels, n_timepoints]))
    temp_labels = np.zeros([nSamples]) 
    event_dic = {'0': 1, '11': 2, '12': 3, '13': 4, '51': 5, '52': 6}
    for i in range(nSamples):
        #temp_features[i][:,:] = features_arr[i][0] #-0.2-1.5s
        temp_features[i][:,:] = features_arr[i][0][:,100:]  #-0.-1.5s
        #temp_features[i][:,:] = features_arr[i][0][:,100:601]  #-0.-1.0s
        #temp_features[i][:,:] = features_arr[i][0][:,:601]  #-0.2-1.0s
        #temp_features[i][:,:] = features_arr[i][0][:,350:] #0.5-1.5s
        #temp_features[i][:,:] = features_arr[i][0][:,600:] #1.0-1.5s
        temp_labels[i] = event_dic[str(label_arr[i][0])]
    #X = np.array(temp_features)
    X = np.array(temp_features)*1000
    y = np.array(temp_labels)
else:
    X, y = preprocess(raw_fname)
    with open(X_path, 'wb') as fx:
        pickle.dump(X, fx)
    with open(y_path, 'wb') as fy:
        pickle.dump(y, fy)
        
print(X.shape,y.shape)
#kernels, chans, samples = 1, 60, 851 #-0.2-1.5s
kernels, chans, samples = 1, 60, 751 #-0.-1.5s
#kernels, chans, samples = 1, 60, 501 #-0.-1.s;0.5-1.5s
#kernels, chans, samples = 1, 60, 251 #1.0-1.5s
#kernels, chans, samples = 1, 60, 601 #-0.2-1.s

# convert labels to one-hot encodings.
y = np_utils.to_categorical(y-2)

# take 50/25/25 percent of the data to train/validate/test
split_factor = 0.75
#X_test       = X[int(X.shape[0]*split_factor):,]
#Y_test       = y[int(X.shape[0]*split_factor):]
X_test       = X[0]     #0 -- 2; 2-- 3 134 132; 6 -- 4
Y_test       = y[0]
#print(np.argwhere(y==3))
#X_test = X[np.argwhere(y==3)]
#Y_test = y[y==3]

# convert data to NHWC (trials, channels, samples, kernels) format. 
X_test       = X_test.reshape(1, chans, samples, kernels)
print(X_test.shape[0], 'test samples')

model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, 
               dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')

# load optimal weights
model.load_weights(CKP_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
print(probs,preds)
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))

#取某一层的输出为输出新建为model，采用函数模型
#dense1_layer_model = Model(inputs=model.input,outputs=model.get_layer('Dense_1').output)
#dense1_output = dense1_layer_model.predict(data)
#print (dense1_output.shape)

#获得某一层的权重和偏置
weight_Depthwise = model.get_layer('depthwise_conv2d').get_weights()
output_Depthwise = model.get_layer('depthwise_conv2d').output
print("<<<<<<<<<,")
#print(weight_Depthwise)
weight_Depthwise = np.array(weight_Depthwise)
print(weight_Depthwise.shape, output_Depthwise.shape)   #shape:(1, 60, 1, 8, 2) (None, 1, 751, 16)

#画地形图
#设置通道名
#biosemi_montage = mne.channels.make_standard_montage('biosemi64')
montage = mne.channels.make_standard_montage("standard_1005")
ch_names = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6',
'F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5',
'P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']

#创建info对象
info = mne.create_info(ch_names=ch_names, sfreq=200.,ch_types='eeg')
#生成数据
data = np.random.randn(60,1)
#创建evokeds对象                       
evoked = mne.EvokedArray(data, info)
#print(evoked.info)
#evoked.set_montage(biosemi_montage)
# 读取正确的导联名称
#new_chan_names = np.loadtxt('Xiaoqing60_AF7.txt', dtype=str, usecols=-1)
#print(new_chan_names)
# 读取电极位置信息
custom__montage = mne.channels.read_custom_montage('Xiaoqing60_AF7.txt')
#evokeds设置通道
#evoked.set_montage(custom__montage)
#evoked.set_montage(montage)
#画图
#mne.viz.plot_topomap(evoked.data[:, 0], evoked.info, show=True)
#plt.show()
#plt.clf()

with DeepExplain(session = K.get_session()) as de:
    #model.load_weights(CKP_PATH)
    input_tensor = model.layers[0].input
    #print(model.layers[-2].output,model.layers[-2].output.shape, Y_test.shape)
    #for i in range(len(model.layers)):
    #    print('<<<<<< layer ',i," ",model.layers[i].input.shape, model.layers[i].output.shape)
    fModel = Model(inputs = input_tensor, outputs = model.layers[-2].output)
    #fModel = Model(inputs = input_tensor, outputs = model.layers[3].output)
    target_tensor = fModel(input_tensor)
    #X_test_true = X_test[np.argwhere(preds == Y_test.argmax(axis=-1))]
    #Y_test_true = Y_test[np.argwhere(preds == Y_test.argmax(axis=-1))]
    print(target_tensor.shape, (target_tensor * Y_test).shape, input_tensor.shape, X_test.shape)
    attributions = de.explain('deeplift', target_tensor * Y_test, input_tensor, X_test)
    #attributions = de.explain('deeplift', target_tensor * Y_test_true, input_tensor, X_test_true)
    #attributions = de.explain('elrp', target_tensor * Y_test, input_tensor, X_test)
    #attributions = de.explain('saliency', T, X, xs[i], ys=ys[i], batch_size=3)
    #attributions = de.explain('occlusion', target_tensor * Y_test, input_tensor, X_test)

# plot the confusion matrices for both classifiers
#names = ['JG', 'MM', 'JY']
#plt.figure(0)
#plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
#plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-4,2')
#plt.show()

# plot the explain
#print(attributions)
print(attributions.shape)
plt.imshow(attributions[0, :, :].squeeze(),aspect='auto')
plt.colorbar()
x=[0,250,500,750]
values = [i/500 for i in x]
plt.xticks(x,values)
plt.xlabel('Time (seconds)')
plt.ylabel('Channels')
plt.show()
plt.clf()
