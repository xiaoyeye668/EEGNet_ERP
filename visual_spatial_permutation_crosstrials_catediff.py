
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
CKP_PATH = './save_models_cross/checkpoint_intents_k125_82_17s_split90_batch16_noscale.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_2ndPooling2.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_2ndPooling4.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_1stPooling16_2ndPooling4.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_1stPooling16_2ndPooling2.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_52_17s_batch16_scale1000_1stPooling16_2ndPooling4.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_1stPooling16_2ndPooling3.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_sublabel_classw_k250_42_17s_batch16_1stPooling16_2ndPooling3.h5'

#Load test dataset
test_titles = ['JG', 'MM', 'JY']
#feature_path = './datasets_cross/datasets_cross_13s/'
feature_path = './datasets_cross/datasets_cross_17s/'
feature_path = './datasets_cross_sublabel/datasets_cross_17s/'


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
               dropoutRate = 0.25, kernLength = 250, F1 = 4, D = 2, F2 = 8, 
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
#X_t,Y_t = X_train[:],Y_train[:]
#probs_t = model.predict(X_t)
#preds_t = probs_t.argmax(axis = -1)  
#print('调用函数auc：', metrics.roc_auc_score(preds_t, Y_t, multi_class='ovr', average='weighted'))

chance_probs = np.full((571, ), 0.33)
#print(chance_probs,chance_probs.shape)
#print(probs.max(axis = -1), probs.max(axis = -1).shape)
stat, p = wilcoxon(probs.max(axis = -1) , chance_probs)
print('stat=%.4f, p=%.4f' % (stat, p))
if p < 0.05/571:
    print("显著差异")
dense_layer = model.get_layer(name='depthwise_conv2d')
#dense_layer = model.get_layer(index=-1)
print(dense_layer.name)
for weight in dense_layer.weights:
    print(weight.name, weight.shape)
    filter_weight = np.array(weight)

#filter_weight = filter_weight.reshape(4,60,1)
#filter_weight = filter_weight.reshape(8,60,2)
filter_weight = filter_weight.reshape(4,60,2)
#filter_weight = filter_weight.reshape(5,60,2)
print(filter_weight.shape)

X,Y = X_train[:],Y_train[:]
print('<<<<<<<< X ', X.shape)

#情绪分类数据
feature_path = './datasets_cross_sublabel/datasets_cross_17s_emotion/'
X_train_e       = np.load(feature_path+ '/' +'X_train.npy')
Y_train_e       = np.load(feature_path+ '/' +'Y_train.npy')

X_train_e       = X_train_e.reshape(X_train_e.shape[0], chans, samples, kernels)
X_e,Y_e = X_train_e[:],Y_train_e[:]

print('<<<<<<<<  X_train_e shape ', X_train_e.shape)
from tensorflow.keras.models import Model
from mne.time_frequency import tfr_array_morlet
#from neurora.stuff import clusterbased_permutation_2d_1samp_2sided, clusterbased_permutation_2d_2sided
from stuff import clusterbased_permutation_2d_2sided, clusterbased_permutation_2d_1samp_2sided
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
    #tfr = np.zeros([len(class_out_data), 1, 21, 651])
    tfr = np.zeros([len(class_out_data), 1, 15, samples+100])
    #tfr = np.zeros([len(class_out_data), 1, 18, samples])

    for i in range(len(class_out_data)):
        #out shape:(None, 1, 851, 16)
        data = class_out_data[i,:,:,0:1].reshape((1,1,samples)) #(n_epochs, n_chans, n_times)
        # 设定一些时频分析的参数
        # 频段选取0.1-32Hz
        freqs = np.arange(2, 32, 2)
        n_cycles = np.array([1]+[i for i in np.arange(4, 21, 2)/3]+[8]*(int(freqs[-1]-21)//2+1))
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

freqs = np.arange(2, 32, 2)
#times = np.arange(400, 1400, 2)
#freqs = np.arange(4, 40, 2)
times = np.arange(200, 1200, 2)
#times = np.arange(-200, 1500, 2)
print('<<<<<<<<<<<<<<<')
print(freqs.shape, times.shape)

class_JG_out, class_JG_label = depthwise_output_byclass(0, X, Y, model)
class_MM_out, class_MM_label = depthwise_output_byclass(1, X, Y, model)
class_JY_out, class_JY_label = depthwise_output_byclass(2, X, Y, model)
tfr_JG = tfr_byclass(class_JG_out, samples)
tfr_MM = tfr_byclass(class_MM_out, samples)
tfr_JY = tfr_byclass(class_JY_out, samples)


class_NG_out, class_NG_label = depthwise_output_byclass(0, X_e, Y_e, model)
class_NE_out, class_NE_label = depthwise_output_byclass(1, X_e, Y_e, model)
class_PO_out, class_PO_label = depthwise_output_byclass(2, X_e, Y_e, model)
tfr_NG = tfr_byclass(class_NG_out, samples)
tfr_NE = tfr_byclass(class_NE_out, samples)
tfr_PO = tfr_byclass(class_PO_out, samples)


def plot_tfr_diff_results(tfr1, tfr2, freqs, times, p=0.01, clusterp=0.05, 
                          threshold=6.0, clim=[-2, 2]):
    
    """
    参数：
        tfr1 : shape为[n_subs, n_freqs, n_times]的矩阵，代表条件1下时频分析的结果
        tfr2 : shape为[n_subs, n_freqs, n_times]的矩阵，代表条件2下时频分析的结果
        freqs : shape为[n_freqs]的array，代表时频分析的频率（对应y轴的频率范围及频率点）
        times : shape为[n_times]的array，代表时频分析的时间点（对应x轴的时间范围及时间点）
        p : 一个浮点型数字，默认为0.01，代表p值的阈值
        clusterp : 一个浮点型数字，默认为0.05，代表cluster层面p值的阈值
        clim : 一个List或array，[最小值，最大值]，默认为[-2, 2]，代表颜色条的上下边界
    """
    
    n_freqs = len(freqs)
    n_times = len(times)
    
    # 统计分析
    # 注意：由于进行了cluster-based permutation test，需要运行较长时间
    # 这里使用NeuroRA的stuff模块下的clusterbased_permutation_2d_2sided()函数
    # 其返回的stats_results为一个shape为[n_freqs, n_times]的矩阵
    # 该矩阵中不显著的点的值为0，条件1显著大于条件2的点的值为1，条件1显著小于条件2的点的值为-1
    # 这里iter设成100是为了示例运行起来快一些，建议1000
    #print(tfr1.shape,tfr2.shape)
    stats_results, cluster_n1, cluster_n2 = clusterbased_permutation_2d_2sided(tfr1, tfr2, 
                                                       p_threshold=p, 
                                                       clusterp_threshold=clusterp,
                                                       threshold=threshold,
                                                       iter=1000)
                                                       
    # 计算△tfr
    tfr_diff = tfr1 - tfr2
    
    # 时频分析结果可视化
    fig, ax = plt.subplots(1, 1)
    # 勾勒显著性区域
    #padsats_results = np.zeros([n_freqs + 2, n_times + 2])
    #padsats_results[1:n_freqs + 1, 1:n_times + 1] = stats_results
    padsats_results = np.zeros([n_freqs, n_times + 2])
    padsats_results[:n_freqs, 1:n_times + 1] = stats_results
    x = np.concatenate(([times[0]-1], times, [times[-1]+1]))
    #y = np.concatenate(([freqs[0]-1], freqs, [freqs[-1]+1]))
    y = freqs
    X, Y = np.meshgrid(x, y)
    #print(np.where(padsats_results==1)[0], np.where(padsats_results==-1)[0])
    if len(np.where(padsats_results>0.5)[0]) > 0:
        ax.contour(X, Y, padsats_results, [0.5], colors="maroon", alpha=1.0, 
               linewidths=2, linestyles="dashed")

    if len(np.where(padsats_results<-0.5)[0]) > 0:
        ax.contour(X, Y, padsats_results, [-0.5], colors="darkblue", alpha=1.0,
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
    print(np.average(tfr_diff, axis=0).shape)

tfr1 = tfr_JG[:500,0,:, 250:750].squeeze()   #补后 0.2-1.2s
tfr2 = tfr_MM[:500,0,:, 250:750].squeeze()
tfr3 = tfr_JY[:500,0,:, 250:750].squeeze()


tfr1_e = tfr_NG[:500,0,:, 250:750].squeeze()   #补后 0.2-1.2s
tfr2_e = tfr_NE[:500,0,:, 250:750].squeeze()
tfr3_e = tfr_PO[:500,0,:, 250:750].squeeze()


def plot_cate_tfr_diff_results(tfr1, tfr2, tfr1_e, tfr2_e, freqs, times, p=0.01, clusterp=0.05, 
                          threshold=6.0, p_e=0.01, clusterp_e=0.05, threshold_e=6.0, clim=[-2, 2]):
    n_freqs = len(freqs)
    n_times = len(times)
    stats_results, cluster_n1, cluster_n2 = clusterbased_permutation_2d_2sided(tfr1, tfr2, 
                                                       p_threshold=p, 
                                                       clusterp_threshold=clusterp,
                                                       threshold=threshold,
                                                       iter=1000)
    stats_results_e, cluster_n1_e, cluster_n2_e = clusterbased_permutation_2d_2sided(tfr1_e, tfr2_e, 
                                                       p_threshold=p_e, 
                                                       clusterp_threshold=clusterp_e,
                                                       threshold=threshold_e,
                                                       iter=1000)                                                   
    # 计算△tfr
    tfr_diff = tfr1 - tfr2
    tfr_diff_e = tfr1_e - tfr2_e
    # 时频分析结果可视化
    fig, ax = plt.subplots(1, 1)
    # 勾勒显著性区域
    #padsats_results = np.zeros([n_freqs + 2, n_times + 2])
    #padsats_results[1:n_freqs + 1, 1:n_times + 1] = stats_results
    padsats_results = np.zeros([n_freqs, n_times + 2])
    padsats_results_e = np.zeros([n_freqs, n_times + 2])
    padsats_results[:n_freqs, 1:n_times + 1] = stats_results
    padsats_results_e[:n_freqs, 1:n_times + 1] = stats_results_e

    print(padsats_results.shape, padsats_results_e.shape)
    x = np.concatenate(([times[0]-1], times, [times[-1]+1]))
    #y = np.concatenate(([freqs[0]-1], freqs, [freqs[-1]+1]))
    y = freqs
    X, Y = np.meshgrid(x, y)
    print(padsats_results,padsats_results_e)
    #print(np.where(padsats_results==1)[0], np.where(padsats_results==-1)[0])
    if len(np.where(padsats_results>0.5)[0]) > 0 and len(np.where(padsats_results_e>0.5)[0]) > 0:
    #if len(np.where(padsats_results_e>0.5)[0]) > 0:
        ax.contour(X, Y, padsats_results + padsats_results_e, [1.5], colors="maroon", alpha=1.0, 
               linewidths=2, linestyles="dashed")

    if len(np.where(padsats_results<-0.5)[0]) > 0 and len(np.where(padsats_results_e<-0.5)[0]) > 0:
#    if len(np.where(padsats_results_e<-0.5)[0]) > 0:
        ax.contour(X, Y, padsats_results + padsats_results_e, [-1.5], colors="darkblue", alpha=1.0,
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

#plot_tfr_diff_results(tfr1, tfr2, freqs, times, 
#                     p=0.0165, clusterp=0.05, threshold=200.0,clim=[-2, 2])
#plot_tfr_diff_results(tfr1_e, tfr2_e, freqs, times, 
#                     p=0.0165, clusterp=0.05, threshold=200.0,clim=[-2, 2])
plot_cate_tfr_diff_results(tfr1, tfr2, tfr1, tfr3, freqs, times, p=0.0165, clusterp=0.05, 
                          threshold=200.0, p_e=0.025, clusterp_e=0.05, threshold_e=60.0, clim=[-2, 2])
#plot_cate_tfr_diff_results(tfr3_e, tfr2_e, tfr3_e, tfr1_e, freqs, times, p=0.0165, clusterp=0.05, 
#                          threshold=200.0, p_e=0.0165, clusterp_e=0.05, threshold_e=200.0, clim=[-2, 2])
#plot_tfr_diff_results(tfr3, tfr2, freqs, times, 
#                      p=0.0165, clusterp=0.05, threshold=200.0,clim=[-2, 2])
#plot_tfr_diff_results(tfr1, tfr3, freqs, times, 
#                      p=0.025, clusterp=0.05, threshold=100.0,clim=[-2, 2])

