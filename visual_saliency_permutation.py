
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
from pyriemann.utils.viz import plot_confusion_matrix

from tf_keras_vis.utils import num_of_gpus
_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)
K.set_image_data_format('channels_last')

#CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_63_13s_batch16_scale1000.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_17s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_2ndPooling2.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_2ndPooling4.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_1stPooling16_2ndPooling4.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_42_17s_batch16_scale1000_1stPooling16_2ndPooling2.h5'
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

#kernels, chans, samples = 1, 60, 751 #0-1.5s
#kernels, chans, samples = 1, 60, 651 #0.2-1.5s
kernels, chans, samples = 1, 60, 851 #-0.2-1.5s
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
#print(probs[150:200], Y_test.argmax(axis=-1)[150:200])
acc = np.mean(preds == Y_test.argmax(axis=-1))
#0-10;195;399
#print(Y_test.argmax(axis=-1)[462],Y_test.argmax(axis=-1)[36],Y_test.argmax(axis=-1)[463])
print("Classification accuracy: %f " % (acc))
print('调用函数auc：', metrics.roc_auc_score(preds, Y_test, multi_class='ovr', average='weighted'))

'''
Saliency Map
'''
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from neurora.stuff import clusterbased_permutation_2d_1samp_2sided, clusterbased_permutation_2d_2sided
import matplotlib.ticker as ticker
# from tf_keras_vis.utils import normalize

#Implement functions required to use attentions
#model modifier
def model_modifier_function(cloned_model):
    #cloned_model.layers[-1].activation = tf.keras.activations.linear
    cloned_model.layers[-2].activation = tf.keras.activations.linear
replace2linear = model_modifier_function(model)

#score function
from tf_keras_vis.utils.scores import CategoricalScore
#score = CategoricalScore([0, 1, 2])
score = CategoricalScore([0])
#score = CategoricalScore([1])
#score = CategoricalScore([2])
print('<<<<<<< output ', model.layers[-2].output, model.layers[-2].output.shape)
#score = score_function(output = model.layers[-2].output)

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)

# Generate saliency map with smoothing that reduce noise by adding noise

X,Y = X_train[150:200],Y_train[150:200]
#X = np.asarray([test_1, test_2, test_3])
#Y = np.asarray([Y_train[3098], Y_train[3040], Y_train[2193]])
print('<<<<<<<< X ', X.shape)
probs = model.predict(X)
preds = probs.argmax(axis = -1) 
probs = probs.max(axis = -1)   
print(probs.shape,preds.shape)
names = ['JG', 'MM', 'JY']
plt.figure(0)
plot_confusion_matrix(preds, Y.argmax(axis = -1), names)
plt.show()
print('调用函数auc：', metrics.roc_auc_score(preds, Y, multi_class='ovr', average='weighted'))


def attribution_byclass(class_index, x, y, score):
    index = np.argwhere(y.argmax(axis=-1)==class_index)[:,:1]
    print(index.shape)
    data, label = x[index].reshape(index.shape[0], chans, samples, kernels), y[index].squeeze()
    print(data.shape, label.shape)
    attribution = saliency(score,
                        data,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20, # noise spread level.   
                        keepdims=True)  #whether or not to keep the channels-dimension. 
    print(attribution.shape)
    return attribution


## Since v0.6.0, calling `normalize()` is NOT necessary.
# saliency_map = normalize(saliency_map)

def plot_attribution_results(attribution, channels, ch_names, times, p=0.01, clusterp=0.05):
    
    n_channels = len(channels)
    n_times = len(times)
    # 统计分析
    # 注意：由于进行了cluster-based permutation test，需要运行较长时间
    # 这里使用NeuroRA的stuff模块下的clusterbased_permutation_2d_1samp_2sided()函数
    # 其返回的stats_results为一个shape为[n_channels, n_times]的矩阵
    # 该矩阵中不显著的点的值为0，显著大于0的点的值为1，显著小于0的点的值为-1
    stats_results = clusterbased_permutation_2d_1samp_2sided(attribution, 0, 
                                                        p_threshold=p,
                                                        clusterp_threshold=clusterp,
                                                        iter=1000)

    # 分析结果可视化
    fig, ax = plt.subplots(1, 1)
    # 勾勒显著性区域
    padsats_results = np.zeros([n_channels + 2, n_times + 2])
    print(padsats_results.shape,stats_results.shape)
    padsats_results[1:n_channels + 1, 1:n_times + 1] = stats_results
    #padsats_results[:n_freqs, :n_times + 1] = stats_results
    x = np.concatenate(([times[0]-1], times, [times[-1]+1]))
    y = np.concatenate(([channels[0]-1], channels, [channels[-1]+1]))
    #y = channels
    X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, padsats_results, [0.5], colors="red", alpha=0.9, 
               linewidths=2, linestyles="dashed")
    ax.contour(X, Y, padsats_results, [-0.5], colors="blue", alpha=0.9,
               linewidths=2, linestyles="dashed")
    # 绘制结果热力图
    im = ax.imshow(np.average(attribution, axis=0), cmap='jet')
    ax.set_aspect('auto')
    cbar = fig.colorbar(im)
    #cbar.set_label('dB')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Channels')
    show_times = np.linspace(0, 600, 5)
    values = [(i/500+0.2)*1000 for i in show_times]
    ax.set_xticks(show_times, values) 
    ax.set_yticks(channels, ch_names)
    plt.show()

def plot_attribution_results_2(attribution, channels, ch_names): 
    # 分析结果可视化
    fig, ax = plt.subplots(1, 1)
    #im = ax.imshow(np.average(attribution, axis=0), cmap='RdYlBu_r', origin='lower', 
    #print(attribution[0],attribution[0].shape)
    im = ax.imshow(np.average(attribution, axis=0), cmap='jet')
    print(attribution.shape)
    ax.set_aspect('auto')
    cbar = fig.colorbar(im)
    #cbar.set_label('dB')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Channels')
    #ax.set_xlim((times[0],times[-1]))
    #x = ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    times = np.linspace(0, 500, 5)
    values = [(i/500+0.2)*1000 for i in times]
    ax.set_xticks(times, values) 
    ax.set_yticks(channels, ch_names)
    plt.show()


def plot_attribution_diff_results(attribution1, attribution2, channels, ch_names, times, p=0.01, clusterp=0.05):
    n_channels = len(channels)
    n_times = len(times)
    
    # 统计分析
    # 注意：由于进行了cluster-based permutation test，需要运行较长时间
    # 这里使用NeuroRA的stuff模块下的clusterbased_permutation_2d_2sided()函数
    # 其返回的stats_results为一个shape为[n_channels, n_times]的矩阵
    # 该矩阵中不显著的点的值为0，条件1显著大于条件2的点的值为1，条件1显著小于条件2的点的值为-1
    # 这里iter设成100是为了示例运行起来快一些，建议1000
    print(attribution1.shape,attribution2.shape)
    stats_results = clusterbased_permutation_2d_2sided(attribution1, attribution2, 
                                                       p_threshold=p,
                                                       clusterp_threshold=clusterp,
                                                       iter=1000)
    
    # 计算△attribution
    attribution_diff = attribution1 - attribution2
    
    # 时频分析结果可视化
    fig, ax = plt.subplots(1, 1)
    # 勾勒显著性区域
    padsats_results = np.zeros([n_channels + 2, n_times + 2])
    padsats_results[1:n_channels + 1, 1:n_times + 1] = stats_results
    x = np.concatenate(([times[0]-1], times, [times[-1]+1]))
    y = np.concatenate(([channels[0]-1], channels, [channels[-1]+1]))
    X, Y = np.meshgrid(x, y)
    ax.contour(X, Y, padsats_results, [0.5], colors="red", alpha=0.9, 
               linewidths=2, linestyles="dashed")
    ax.contour(X, Y, padsats_results, [-0.5], colors="blue", alpha=0.9,
               linewidths=2, linestyles="dashed")
    # 绘制时频结果热力图
    im = ax.imshow(np.average(attribution_diff, axis=0), cmap='jet')
    ax.set_aspect('auto')
    cbar = fig.colorbar(im)
    #cbar.set_label('Attribution weight')
    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('Channels')
    #ax.set_xlim((times[0],times[-1]))
    #x = ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
    show_times = np.linspace(0, 600, 5)
    values = [(i/500+0.2)*1000 for i in show_times]
    ax.set_xticks(show_times, values) 
    ax.set_yticks(channels, ch_names)
    plt.show()


times = np.arange(200, 1200, 2)
#times = np.linspace(0, 600, 5)   # 产生区间在0.2-1.4s间的5个均匀数值
channels = list(range(0,60))
ch_names = np.loadtxt('Xiaoqing60_AF7.txt', dtype=str, usecols=-1)
ch_names = list(ch_names)
ch_names.remove('COMNT')
ch_names.remove('SCALE')
#ch_names.reverse()

#x=[0,250,500,650]
#x=[200,250,500,750]
#values = [i/500+0.2 for i in x]

attribution_JG = attribution_byclass(0, X, Y, score)
#attribution_MM = attribution_byclass(1, X, Y, score)
#attribution_JY = attribution_byclass(2, X, Y, score)
attribution = np.array(attribution_JG[:, :, 200:700]).squeeze()     #0.2-1.2s
#attribution = np.array(attribution_MM[:, :, 200:700]).squeeze()     #0.2-1.2s
#attribution = np.array(attribution_JY[:, :, 200:700]).squeeze()     #0.2-1.2s

#plot_attribution_results(attribution, channels, ch_names, times, p=0.05, clusterp=0.05)
plot_attribution_results_2(attribution, channels, ch_names)
'''
attribution_1 = np.array(attribution_JG[:10, :, 200:800]).squeeze()   #0.2-1.4s
attribution_2 = np.array(attribution_MM[:10, :, 200:800]).squeeze()
#attribution_3 = np.array(attribution_JY[:10, :, 200:800]).squeeze()
diff_attribution = np.average(attribution_1, axis=0) - np.average(attribution_2, axis=0)

plot_attribution_results(diff_attribution[np.newaxis,:], channels, ch_names, times, p=0.025, clusterp=0.05)
#plot_attribution_diff_results(attribution_1, attribution_2, channels, ch_names, times, p=0.025)

for i, title in enumerate(test_titles):
    #plt.subplot(1, 3, i+1)
    plt.figure(figsize=(10, 8))
    #plt.title(title+' {}'.format(probs[i]), fontsize=12)
    plt.title(title+' {}'.format('saliency map'), fontsize=12)
    plt.xticks(x, values)
    plt.yticks(y, ch_names)
    print(saliency_map[i].shape) 
    img = plt.imshow(saliency_map[i], cmap='jet', aspect='auto') 
    plt.colorbar(img)
    plt.show()
#f.colorbar(img)
#plt.show()
'''
