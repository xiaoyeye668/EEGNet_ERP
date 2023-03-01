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

CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_17s_batch16_scale1000.h5'

#Load test dataset
test_titles = ['JG', 'MM', 'JY']
feature_path = './datasets_cross_new/datasets_cross_17s_split90/'

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
               dropoutRate = 0.3, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')
print(model.summary())
# load optimal weights
model.load_weights(CKP_PATH)

# predict accuracy
probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  

dense_layer = model.get_layer(name='depthwise_conv2d')
#dense_layer = model.get_layer(index=-1)
for weight in dense_layer.weights:
    print(weight.name, weight.shape)
    filter_weight = np.array(weight)

#filter_weight = filter_weight.reshape(4,60,1)
filter_weight = filter_weight.reshape(8,60,2)
#filter_weight = filter_weight.reshape(4,60,3)
#filter_weight = filter_weight.reshape(6,60,3)

import mne
from mne import io, pick_types, read_events, Epochs, EvokedArray
event_id = dict(weight=0)
#n_filter = 6
n_filter = 8
#n_filter = 4
fig, axes = plt.subplots(nrows=len(event_id), ncols=n_filter,
                        figsize=(8,4))
                        #figsize=(n_filter, len(event_id) * 2))

ch_names = np.loadtxt('Xiaoqing60_AF7.txt', dtype=str, usecols=-1)
ch_names = list(ch_names)
ch_names.remove('COMNT')
ch_names.remove('SCALE')
ch_names.reverse()
print(len(ch_names))
#ch_names = [str(i) for i in range(1,61)] #通道名称
sfreq = 500 #采样率
montage = mne.channels.make_standard_montage("standard_1020")
#tmp_info = mne.create_info(ch_names=montage.ch_names, sfreq=500, ch_types = "eeg") #创建信号的信息
#print(montage.ch_names)
tmp_info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types = "eeg") #创建信号的信息

print('<<<<<<<<<< filter_weight shape ', filter_weight.shape)
for ii in np.arange(n_filter):
    pattern_evoked = EvokedArray(filter_weight[ii,:,:1].reshape(60,1), tmp_info)
    #print(pattern_evoked.info)
    pattern_evoked.set_montage(montage)
    #pattern_evoked.set_montage(montage,on_missing='warn')
    #pattern_evoked.info['bads'].extend(['FP1', 'FPZ', 'FP2', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']) 
    
    #mne.viz.plot_topomap(pattern_evoked.data[:, 0], pattern_evoked.info,show=False)
    pattern_evoked.plot_topomap(
        times=0.0,
        time_format=' %d' if ii == 0 else '', colorbar=False,
        show_names=False, axes=axes[ii], show=False)
    axes[ii].set(ylabel='spatial filter {}'.format(ii+1))
fig.tight_layout(h_pad=1.0, w_pad=1.0, pad=0.1)
plt.show()

print(dense_layer.output, dense_layer.output.shape, type(dense_layer.output))
#dense_layer_output = np.array(dense_layer.output)
# 0-221 1-3 2-203
test_1, test_2, test_3 = X_train[221], X_train[3], X_train[203]
X = np.asarray([test_1, test_2, test_3])
print('<<<<<<<< X ', X.shape)

from tensorflow.keras.models import Model
from mne.time_frequency import tfr_array_morlet
from neurora.stuff import clusterbased_permutation_2d_1samp_2sided

#取某一层的输出为输出新建为model，采用函数模型
Depthwise_layer_model = Model(inputs=model.input,outputs=model.get_layer('depthwise_conv2d').output)
data = X
Depthwise_layer_output = Depthwise_layer_model.predict(data)
print (Depthwise_layer_output.shape)

tfr = np.zeros([2, 1, 16, 851])
#tfr = np.zeros([2, 1, 21, 651])
#tfr = np.zeros([2, 1, 21, 601])
#tfr = np.zeros([2, 1, 26, 651])
# 逐被试迭代
diff = 3
for classes in np.arange(1,diff):
    print("******** ", classes)
    class_1 = Depthwise_layer_output[classes-1:classes,:,:,:1]
    class_2 = Depthwise_layer_output[classes:classes+1,:,:,:1]
    # 数据的shape从[n_trials, n_channels, n_times,filter_num]转换成[n_trials, n_channels, n_times]
    if classes == 1:
        #tmp_class = class_1
        #continue
        subdata = class_1 - class_2
    else:
        #subdata = class_2 - tmp_class
        subdata = class_2 - class_1
    subdata = subdata.reshape((1,1,851)) #(n_epochs, n_chans, n_times)
    #subdata = subdata.reshape((1,1,651))
    #subdata = subdata.reshape((1,1,601))
    print(subdata.shape)
    
    # 设定一些时频分析的参数
    # 频段选取0.1-40Hz
    freqs = np.arange(0.1, 32, 2)
    n_cycles = np.array([i for i in np.arange(0.1, 20, 2)/3]+[8]*(int(freqs[-1]-20)//2+1))
  
    # 时频分析
    # 使用MNE的time_frequency模块下的tfr_arrayy_morlet()函数
    # 其输入为[n_epochs, n_channels, n_times]的array
    # 同时接着依次传入数据采样率、计算频率、周期数和输出数据类型
    subtfr = tfr_array_morlet(subdata, 500, freqs, n_cycles, output='power')

    # 此时返回的tfr的shape为[n_trials, n_channels, n_freqs, n_times],(139, 60, 21, 851)
    # 这里，对试次与导联维度平均传入tfr变量中
    tfr[classes-1] = np.average(subtfr, axis=0)
    # 基线校正，这里使用'logratio'方法，即除以基线均值并取log
    # 取基线为-200到0ms
    for chl in range(1):    
        for freq in range(len(freqs)):
            tfr[classes-1,chl,freq] = 10 * np.log10(tfr[classes-1, chl, freq] / 
                                np.average(tfr[classes-1, chl, freq, :200]))
    print('<<<<<<<<<<<<<<< tfr ', tfr.shape)

def plot_tfr_results_2(tfr, freqs, times, clim=[-4, 8]):
    
    n_freqs = len(freqs)
    n_times = len(times)
    
    # 时频分析结果可视化
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.average(tfr, axis=0), cmap='RdYlBu_r', origin='lower', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1]], clim=clim)
    ax.set_aspect('auto')
    cbar = fig.colorbar(im,ticks=[0])
    #cbar.set_label('dB')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    plt.show()

freqs = np.arange(0.1, 32, 2)
times = np.arange(0, 1500, 4)
print(freqs.shape, times.shape)

show_list = [1,2]
for i in show_list:
    tfr_diff = tfr[i-1:i, 0, :, 100:]
    plot_tfr_results_2(tfr_diff, freqs, times, clim=[-5, 5])