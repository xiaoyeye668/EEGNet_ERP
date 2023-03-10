
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
import scipy.io as sio

from tf_keras_vis.utils import num_of_gpus
_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)
K.set_image_data_format('channels_last')

CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_17s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_41_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_43_12s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k125_82_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_63_13s_batch16_scale1000.h5'


#Load test dataset
test_titles = ['JG', 'MM', 'JY']
#feature_path = './datasets_cross/datasets_cross_13s/'
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
               dropoutRate = 0.3, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
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
X_t,Y_t = X_train[:],Y_train[:]
probs_t = model.predict(X_t)
preds_t = probs_t.argmax(axis = -1)  
print('调用函数auc：', metrics.roc_auc_score(preds_t, Y_t, multi_class='ovr', average='weighted'))

chance_probs = np.full((509, ), 0.33)
#print(chance_probs,chance_probs.shape)
#print(probs.max(axis = -1), probs.max(axis = -1).shape)
stat, p = wilcoxon(probs.max(axis = -1) , chance_probs)
print('stat=%.4f, p=%.4f' % (stat, p))

dense_layer = model.get_layer(name='depthwise_conv2d')
#dense_layer = model.get_layer(index=-1)
print(dense_layer.name)
for weight in dense_layer.weights:
    print(weight.name, weight.shape)
    filter_weight = np.array(weight)

#filter_weight = filter_weight.reshape(4,60,1)
filter_weight = filter_weight.reshape(8,60,2)
#filter_weight = filter_weight.reshape(4,60,3)
#filter_weight = filter_weight.reshape(6,60,3)
print(filter_weight.shape)

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

tmp_info = mne.create_info(ch_names=ch_names, sfreq=500, ch_types = "eeg") #创建信号的信息

for ii in np.arange(1,n_filter+1):
    pattern_evoked = EvokedArray(filter_weight[ii-1:ii,:,1:2].reshape(60,1), tmp_info)
    #print(pattern_evoked.info)
    pattern_evoked.set_montage(montage)
    #pattern_evoked.set_montage(montage,on_missing='warn')
    #pattern_evoked.info['bads'].extend(['FP1', 'FPZ', 'FP2', 'FZ', 'FCZ', 'CZ', 'CPZ', 'PZ', 'POZ', 'OZ']) 
    
    #mne.viz.plot_topomap(pattern_evoked.data[:, 0], pattern_evoked.info,show=False)
    pattern_evoked.plot_topomap(
        times=0.0,
        time_format=' %d' if ii == 0 else '', colorbar=False,
        show_names=False, axes=axes[ii-1], show=False)
    axes[ii-1].set(ylabel='spatial filter {}'.format(ii))
fig.tight_layout(h_pad=1.0, w_pad=1.0, pad=0.1)
plt.show()

#print(dense_layer.output, dense_layer.output.shape, type(dense_layer.output))
#dense_layer_output = np.array(dense_layer.output)
# 0-3098;2595 1-3040 2-2193
test_1, test_2, test_3 = X_train[3098], X_train[3040], X_train[2193]
print(Y_train[3098], Y_train[3040], Y_train[2193])
X = np.asarray([test_1, test_2, test_3])

#TP_index  = np.argwhere(preds == Y_test.argmax(axis=-1))
#X, Y = X_test[TP_index].reshape(TP_index.shape[0], chans, samples, kernels), Y_test[TP_index]
#JG:34; MM:28; JY:38
X,Y = X_train[:],Y_train[:]
print('<<<<<<<< X ', X.shape)


from tensorflow.keras.models import Model
from mne.time_frequency import tfr_array_morlet
from neurora.stuff import clusterbased_permutation_2d_1samp_2sided, clusterbased_permutation_2d_2sided
#print(preds,preds.shape)
#取某一层的输出为输出新建为model，采用函数模型
def depthwise_output_byclass(class_index, x, y, model):
    Depthwise_layer_model = Model(inputs=model.input,outputs=model.get_layer('depthwise_conv2d').output)
    index = np.argwhere(y.argmax(axis=-1)==class_index)[:,:1]
    data, label = x[index].reshape(index.shape[0], chans, samples, kernels), y[index].squeeze()
    Depthwise_layer_output = Depthwise_layer_model.predict(data)
    print (Depthwise_layer_output.shape)
    return Depthwise_layer_output, label

def tfr_byclass(class_out_data, samples):
    #tfr = np.zeros([len(class_out_data), 1, 16, samples+100])
    tfr = np.zeros([len(class_out_data), 1, 15, samples+100])
    #tfr = np.zeros([len(class_out_data), 1, 18, samples])

    for i in range(len(class_out_data)):
        #out shape:(None, 1, 851, 16)
        data = class_out_data[i,:,:,0:1].reshape((1,1,samples)) #(n_epochs, n_chans, n_times)
        # 设定一些时频分析的参数
        # 频段选取0.1-32Hz
        #freqs = np.arange(0.1, 32, 2)
        #n_cycles = np.array([0.05,1]+[i for i in np.arange(4, 18, 2)/3]+[7]*(int(freqs[-1]-18)//2+1))
        freqs = np.arange(2, 32, 2)
        n_cycles = np.array([1]+[i for i in np.arange(4, 21, 2)/3]+[8]*(int(freqs[-1]-21)//2+1))
        #print('<<<<<<<<<')
        #print(freqs,freqs.shape)
        #print(n_cycles,n_cycles.shape)
        # 时频分析
        # 使用MNE的time_frequency模块下的tfr_arrayy_morlet()函数
        # 其输入为[n_epochs, n_channels, n_times]的array
        # 同时接着依次传入数据采样率、计算频率、周期数和输出数据类型
        #分析data用-200ms~-100ms倒置后补-300ms~-200ms数据
        data = np.concatenate((data[:,:,99::-1],data), axis=2)
        subtfr = tfr_array_morlet(data, 500, freqs, n_cycles, output='power')
        # 此时返回的tfr的shape为[n_trials, n_channels, n_freqs, n_times],(1, 1, 16, 851)
        # 这里，对试次与导联维度平均传入tfr变量中
        tfr[i] = subtfr
        # 基线校正，这里使用'logratio'方法，即除以基线均值并取log
        # 取基线为-100到0ms
        # 取基线为300到400ms
        for chl in range(1):    
            for freq in range(len(freqs)):
                tfr[i,chl,freq] = 10 * np.log10(tfr[i, chl, freq] / 
                                  np.average(tfr[i, chl, freq, 100:150]))
    return tfr

def plot_tfr_results(tfr, freqs, times, p=0.01, clusterp=0.05, clim=[-4, 4]):
    
    """
    参数：
        tfr : shape为[n_subs, n_freqs, n_times]的矩阵，代表时频分析的结果
        freqs : shape为[n_freqs]的array，代表时频分析的频率（对应y轴的频率范围及频率点）
        times : shape为[n_times]的array，代表时频分析的时间点（对应x轴的时间范围及时间点）
        p : 一个浮点型数字，默认为0.01，代表p值的阈值
        clusterp : 一个浮点型数字，默认为0.05，代表cluster层面p值的阈值
        clim : 一个List或array，[最小值，最大值]，默认为[-4, 4]，代表颜色条的上下边界
    """
    
    n_freqs = len(freqs)
    n_times = len(times)
    
    # 统计分析
    # 注意：由于进行了cluster-based permutation test，需要运行较长时间
    # 这里使用NeuroRA的stuff模块下的clusterbased_permutation_2d_1samp_2sided()函数
    # 其返回的stats_results为一个shape为[n_freqs, n_times]的矩阵
    # 该矩阵中不显著的点的值为0，显著大于0的点的值为1，显著小于0的点的值为-1
    # 这里iter设成100是为了示例运行起来快一些，建议1000
    stats_results = clusterbased_permutation_2d_1samp_2sided(tfr, 0, 
                                                        p_threshold=p,
                                                        clusterp_threshold=clusterp,
                                                        iter=1000)
    #print('<<<<<<<<<<<stats_results ', stats_results)                                                    
    
    # 时频分析结果可视化
    fig, ax = plt.subplots(1, 1)
    # 勾勒显著性区域
    padsats_results = np.zeros([n_freqs + 2, n_times + 2])
    print(padsats_results.shape,stats_results.shape)
    padsats_results[1:n_freqs + 1, 1:n_times + 1] = stats_results
    #padsats_results[:n_freqs, :n_times + 1] = stats_results
    x = np.concatenate(([times[0]-1], times, [times[-1]+1]))
    y = np.concatenate(([freqs[0]-1], freqs, [freqs[-1]+1]))
    X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, padsats_results, [0.5], colors="red", alpha=0.9, 
               linewidths=2, linestyles="dashed")
    ax.contour(X, Y, padsats_results, [-0.5], colors="blue", alpha=0.9,
               linewidths=2, linestyles="dashed")
    # 绘制时频结果热力图
    im = ax.imshow(np.average(tfr, axis=0), cmap='RdBu_r', origin='lower', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1]], clim=clim)
    ax.set_aspect('auto')
    cbar = fig.colorbar(im)
    #cbar.set_label('dB')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    plt.show()

def plot_tfr_results_2(tfr, freqs, times, clim=[-4, 8]):
    
    n_freqs = len(freqs)
    n_times = len(times)
    
    # 时频分析结果可视化
    fig, ax = plt.subplots(1, 1)
    im = ax.imshow(np.average(tfr, axis=0), cmap='RdYlBu_r', origin='lower', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1]], clim=clim)
    ax.set_aspect('auto')
    cbar = fig.colorbar(im,ticks=[0])
    cbar.set_label('dB')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    plt.show()

freqs = np.arange(2, 32, 2)
#times = np.arange(400, 1400, 2)
#freqs = np.arange(4, 40, 2)
times = np.arange(200, 1400, 2)
#times = np.arange(-200, 1500, 2)
print('<<<<<<<<<<<<<<<')
print(freqs.shape, times.shape)

class_JY_out, class_JY_label = depthwise_output_byclass(2, X, Y, model)
class_MM_out, class_MM_label = depthwise_output_byclass(1, X, Y, model)
class_JG_out, class_JG_label = depthwise_output_byclass(0, X, Y, model)
tfr_JG = tfr_byclass(class_JG_out, samples)     #shape:(1048, 1, 15, 951)
tfr_MM = tfr_byclass(class_MM_out, samples)
tfr_JY = tfr_byclass(class_JY_out, samples)


tfr1 = tfr_JG[:1000,0,:, 250:850].squeeze()   #补后 0.2-1.4s
tfr2 = tfr_MM[:1000,0,:, 250:850].squeeze()
tfr3 = tfr_JY[:1000,0,:, 250:850].squeeze()
mat_path = 'tfr_res/'
print(tfr1.shape, tfr2.shape, tfr3.shape) 
sio.savemat(mat_path+'tfr_JG.mat', {'data': tfr1})
sio.savemat(mat_path+'tfr_MM.mat', {'data': tfr2})
sio.savemat(mat_path+'tfr_JY.mat', {'data': tfr3})

from mne.time_frequency import tfr_multitaper
from mne.stats import permutation_cluster_1samp_test
threshold = 1.0
n_permutations = 1024
tail = 0
t_power = 1  # t统计量的指数

T_obs, clusters, cluster_p_values, _ = permutation_cluster_1samp_test(
    tfr1, n_permutations=n_permutations, threshold=threshold, tail=tail,
    stat_fun=lambda x: np.mean(x, axis=0) / np.std(x, axis=0), n_jobs=1)
good_cluster_inds = np.where(cluster_p_values < 0.05)[0]
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i_clu, clu_idx in enumerate(good_cluster_inds):
    time_inds, freq_inds = np.squeeze(clusters[clu_idx])
    f_map = T_obs[freq_inds, :].mean(axis=0)
    v_max = np.max(np.abs(f_map))

    fig, axs = plt.subplots(1, 3, figsize=(10, 3))
    axs[0].imshow(f_map[np.newaxis, :], cmap='RdBu_r',
                  extent=[times[0], times[-1], freqs[0], freqs[-1]],
                  aspect='auto', origin='lower', vmin=-v_max, vmax=v_max)
    axs[0].set_title('Cluster {0}'.format(i_clu + 1))

'''
# 显示显著性集群
fig, axs = plt.subplots(1, 1, figsize=(12, 8))
mne.viz.plot_compare_evokeds(dict(data_1=power.data[:, :, subjects==1].mean(axis=2), data_2=power.data[:, :, subjects==2].mean(axis=2)), picks=[np.arange(power.data.shape[1])], legend='lower right', show=False, axes=axs)
sig_times = [power.times[x] for x in np.where(p_values < p_accept)[1]]
ymax = axs.get_ylim()[1]
for time in sig_times:
    axs.vlines(x=time, ymin=0, ymax=ymax, linestyles='--', color='red')
'''




"""
#tfr = tfr_JG[:,0,:, 100:600]   #0.4-1.4s #0.3-1.4s
#tfr = tfr_JG[:,0,:, 200:800]       #0.2-1.4s
#tfr = tfr_JG[:,0,:, 100:800]    #-0.-1.4s
tfr = tfr_JG[:,0,:, 250:850]    #补后0.2-1.4s
#plot_tfr_results(tfr, freqs, times, p=0.05, clusterp=0.05, clim=[-3, 3])
#plot_tfr_results_2(tfr, freqs, times, clim=[-2, 2])
tfr = tfr_MM[:,0,:, 250:850]
#plot_tfr_results_2(tfr, freqs, times, clim=[-2, 2])
tfr = tfr_JY[:,0,:, 250:850]
#plot_tfr_results_2(tfr, freqs, times, clim=[-2, 2])



def plot_tfr_diff_results(tfr1, tfr2, freqs, times, p=0.01, clusterp=0.05, 
                          clim=[-2, 2]):
    
    n_freqs = len(freqs)
    n_times = len(times)
    
    # 统计分析
    # 注意：由于进行了cluster-based permutation test，需要运行较长时间
    # 这里使用NeuroRA的stuff模块下的clusterbased_permutation_2d_2sided()函数
    # 其返回的stats_results为一个shape为[n_freqs, n_times]的矩阵
    # 该矩阵中不显著的点的值为0，条件1显著大于条件2的点的值为1，条件1显著小于条件2的点的值为-1
    # 这里iter设成100是为了示例运行起来快一些，建议1000
    #print(tfr1.shape,tfr2.shape)
    stats_results = clusterbased_permutation_2d_2sided(tfr1, tfr2, 
                                                       p_threshold=p,
                                                       clusterp_threshold=clusterp,
                                                       iter=1000)
    
    # 计算△tfr
    tfr_diff = tfr1 - tfr2
    
    # 时频分析结果可视化
    fig, ax = plt.subplots(1, 1)
    # 勾勒显著性区域
    padsats_results = np.zeros([n_freqs + 2, n_times + 2])
    padsats_results[1:n_freqs + 1, 1:n_times + 1] = stats_results
    x = np.concatenate(([times[0]-1], times, [times[-1]+1]))
    y = np.concatenate(([freqs[0]-1], freqs, [freqs[-1]+1]))
    X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, padsats_results, [0.5], colors="red", alpha=0.9, 
               linewidths=2, linestyles="dashed")
    ax.contour(X, Y, padsats_results, [-0.5], colors="blue", alpha=0.9,
               linewidths=2, linestyles="dashed")
    # 绘制时频结果热力图
    im = ax.imshow(np.average(tfr_diff, axis=0), cmap='RdBu_r', origin='lower', 
                   extent=[times[0], times[-1], freqs[0], freqs[-1]], clim=clim)
    ax.set_aspect('auto')
    cbar = fig.colorbar(im)
    cbar.set_label('$\Delta$ dB')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Frequency (Hz)')
    plt.show()

#freqs = np.arange(4, 32, 2)
#times = np.arange(-200, 1000, 4)
#freqs = np.arange(0.1, 32, 2)
#times = np.arange(400, 1300, 2)
#tfr_JG = tfr_byclass(class_JG_out)
#tfr_MM = tfr_byclass(class_MM_out)

#tfr1 = tfr_JG[:70,0,:, 100:600].squeeze()   #0.4-1.4s
#tfr2 = tfr_MM[:,0,:, 100:600].squeeze()
#tfr1 = tfr_JG[:70,0,:, 200:800].squeeze()
#tfr2 = tfr_MM[:70,0,:, 200:800].squeeze()
#tfr1 = tfr_JG[:,0,:, 100:600].squeeze()   #0.4-1.4s
#tfr2 = tfr_MM[:29,0,:, 100:800].squeeze()  0-1.4s
tfr1 = tfr_JG[:1000,0,:, 250:850].squeeze()   #补后 0.2-1.4s
tfr2 = tfr_MM[:1000,0,:, 250:850].squeeze()
tfr3 = tfr_JY[:1000,0,:, 250:850].squeeze()
print(tfr1.shape, tfr2.shape,tfr3.shape)
plot_tfr_diff_results(tfr1, tfr2, freqs, times, 
                      p=0.025, clusterp=0.05, clim=[-2, 2])
plot_tfr_diff_results(tfr3, tfr2, freqs, times, 
                      p=0.025, clusterp=0.05, clim=[-2, 2])
#plot_tfr_diff_results(tfr1, tfr3, freqs, times, 
#                      p=0.05, clusterp=0.025, clim=[-2, 2])

#plot_tfr_results_2(tfr1-tfr2,freqs, times,clim=[-2, 2] )
"""

#import seaborn as sns
#a = np.array([1,2,3])
#sns.boxplot(a)