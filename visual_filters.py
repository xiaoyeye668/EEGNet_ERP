import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
# EEGNet-specific imports
from EEGModels import EEGNet
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K
from sklearn import metrics
from scipy.stats import wilcoxon

from tf_keras_vis.utils import num_of_gpus
_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)
K.set_image_data_format('channels_last')

#CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_17s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k125_82_17s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_2ndPooling2.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_2ndPooling4.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_52_17s_batch16_scale1000_1stPooling16_2ndPooling4.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_1stPooling16_2ndPooling2.h5'

#Load test dataset
test_titles = ['JG', 'MM', 'JY']
feature_path = './datasets_cross/datasets_cross_17s/'

X_train       = np.load(feature_path+ '/' +'X_train.npy')
Y_train       = np.load(feature_path+ '/' +'Y_train.npy')
X_test       = np.load(feature_path+ '/' +'X_test.npy')
Y_test       = np.load(feature_path+ '/' +'Y_test.npy')

kernels, chans, samples = 1, 60, 851 #-0.2-1.5s
#kernels, chans, samples = 1, 60, 751 #0-1.5s
#kernels, chans, samples = 1, 60, 651 #0.2-1.5s
#kernels, chans, samples = 1, 60, 601 #0.-1.2s
X_train       = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
print(X_test.shape[0], 'test samples') 

model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, 
               dropoutRate = 0.25, kernLength = 250, F1 = 5, D = 2, F2 = 10, 
               dropoutType = 'Dropout')
print(model.summary())
# load optimal weights
model.load_weights(CKP_PATH)

# predict accuracy
probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
acc = np.mean(preds == Y_test.argmax(axis=-1))
print("Classification accuracy: %f " % (acc))
print('调用函数auc：', metrics.roc_auc_score(preds, Y_test, multi_class='ovr', average='weighted'))
chance_probs = np.full((509, ), 0.33)
#print(chance_probs,chance_probs.shape)
#print(probs.max(axis = -1), probs.max(axis = -1).shape)
stat, p = wilcoxon(probs.max(axis = -1) , chance_probs)
print('stat=%.4f, p=%.4f' % (stat, p))
if p > 0.05/509:
    print('Probably the same distribution')
else:
    print('Probably different distributions')
#n_filter = 8
#n_filter = 4
n_filter = 5
dense_layer = model.get_layer(name='depthwise_conv2d')
#dense_layer = model.get_layer(index=-1)
print(dense_layer.name)
for weight in dense_layer.weights:
    print(weight.name, weight.shape)
    filter_weight = np.array(weight)

#filter_weight = filter_weight.reshape(4,60,1)
#filter_weight = filter_weight.reshape(8,60,2)
filter_weight = filter_weight.reshape(5,60,2)
#filter_weight = filter_weight.reshape(6,60,3)
print(filter_weight.shape)

import mne
from mne import io, pick_types, read_events, Epochs, EvokedArray
event_id = dict(weight=0)

fig, axes = plt.subplots(nrows=len(event_id), ncols=n_filter,
                        figsize=(8,4))
                        #figsize=(n_filter, len(event_id) * 2))

ch_names = np.loadtxt('Xiaoqing60_AF7.txt', dtype=str, usecols=-1)
ch_names = list(ch_names)
ch_names.remove('COMNT')
ch_names.remove('SCALE')
#ch_names.reverse()
print(len(ch_names))
#ch_names = [str(i) for i in range(1,61)] #通道名称
sfreq = 500 #采样率
montage = mne.channels.make_standard_montage("standard_1020")
montage = mne.channels.read_custom_montage('Xiaoqing60_AF7.xyz')

#info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types = "eeg") #创建信号的信息
tmp_info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types = "eeg") #创建信号的信息

for ii in np.arange(1,n_filter+1):
    #pattern_evoked = EvokedArray(filter_weight[ii-1:ii].reshape(60,1), tmp_info, tmin=0)
    pattern_evoked = EvokedArray(filter_weight[ii-1,:,1:2].reshape(60,1), tmp_info)   
    #print(pattern_evoked.info)
    #pattern_evoked.plot_sensors()
    pattern_evoked.set_montage(montage)
    
    pattern_evoked.plot_topomap(
        times=0.0,
        time_format=' %d' if ii == 0 else '', colorbar=False,
        show_names=False, axes=axes[ii-1], show=False)
        #show_names=False, axes=axes[ii-1], show=False,  outlines='head',sphere='auto')
    axes[ii-1].set(ylabel='spatial filter {}'.format(ii))
fig.tight_layout(h_pad=1.0, w_pad=1.0, pad=0.1)
plt.show()

print(dense_layer.output, dense_layer.output.shape, type(dense_layer.output))

"""
temporal filter
"""
from scipy.ndimage import gaussian_filter1d

dense_layer = model.get_layer(name='conv2d')
#dense_layer = model.get_layer(index=-1)
print(dense_layer.name)
for weight in dense_layer.weights:
    print(weight.name, weight.shape)
    filter_weight = np.array(weight)

kernLength , fs = 250, 500
#temp_filter_weight = filter_weight.reshape(n_filter,kernLength)
temp_filter_weight = filter_weight.reshape(n_filter,kernLength)

#f, ax = plt.subplots(nrows=1, ncols=4, figsize=(12, 4))
f, ax = plt.subplots(nrows=1, ncols=5, figsize=(12, 4))
#f, ax = plt.subplots(nrows=2, ncols=3, figsize=(12, 8))
x = np.linspace(0, kernLength/fs, kernLength)

for i in np.arange(1,n_filter+1):
    ax[i-1].set_title(i, fontsize=8)
    ax[i-1].set_xlim(0,kernLength/fs)
    #print(temp_filter_weight[i-1:i,:].shape)
    y = temp_filter_weight[i-1:i].squeeze()
    y_smoothed = gaussian_filter1d(y, sigma=6)
    ax[i-1].set_ylim(-0.15, 0.15)
    ax[i-1].plot(x, y_smoothed, color='blue', linewidth=1)
    #ax[i-1].plot(x, temp_filter_weight[i-1:i].squeeze(), color='blue', linewidth=1)
    #ax[i].axis('off')

'''
for i in np.arange(1,3):
    for j in np.arange(1,5):
    #for j in np.arange(1,4):
        ax[i-1][j-1].set_title((i-1)*4+j, fontsize=8)
        #ax[i-1][j-1].set_title((i-1)*3+j, fontsize=8)
        ax[i-1][j-1].set_xlim(0,kernLength/fs)
        #print(temp_filter_weight[i-1:i,:].shape)
        y = temp_filter_weight[(i-1)*4+j-1:(i-1)*4+j].squeeze()
        #y = temp_filter_weight[(i-1)*3+j-1:(i-1)*3+j].squeeze()
        y_smoothed = gaussian_filter1d(y, sigma=1)
        ax[i-1][j-1].set_ylim(-0.15, 0.15)
        ax[i-1][j-1].plot(x, y_smoothed, color='blue', linewidth=1)
'''
plt.tight_layout()
plt.show()
