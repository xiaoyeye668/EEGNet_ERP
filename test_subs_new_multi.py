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
from sklearn import metrics
from scipy.stats import wilcoxon

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)
K.set_image_data_format('channels_last')

#模型保存路径
'''-0.2-1.5s'''
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_17s_batch16_scale1000.h5'

'''0-1.2s'''
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_12s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k250_43_12s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k125_43_12s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k125_82_12s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_43_12s_batch16_scale1000.h5'

'''0-1.5s'''
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_15s_batch16_scale1000.h5'

'''0.2-1.2s'''
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_43_1s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_1s_batch16_scale1000.h5'

'''0.2-1.5s'''
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k25_82_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k12_82_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k5_82_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_81_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_41_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k64_43_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k125_82_13s_batch16_scale1000.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_k250_63_13s_batch16_scale1000.h5'
#CKP_PATH = './save_models_cross/checkpoint_intents_k125_62_13s_batch16_scale1000.h5'
CKP_PATH = './save_models_cross/checkpoint_intents_sublabel_classw_k250_42_17s_batch16_1stPooling16_2ndPooling3.h5'

#feature_path = './datasets_cross/datasets_cross_17s/'
#feature_path = './datasets_cross_sublabel/datasets_cross_17s_emotion/'
feature_path = './datasets_cross_sublabel/datasets_cross_17s/'

X_test       = np.load(feature_path+ '/' +'X_test.npy')
Y_test       = np.load(feature_path+ '/' +'Y_test.npy')
# 0-3098;2595 1-3040 2-2193
#X_test       = np.load(feature_path+ '/' +'X_train.npy')[2800:2900]
#Y_test       = np.load(feature_path+ '/' +'Y_train.npy')[2800:2900]
print(X_test.shape[0], 'test samples')
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

'''
#数据随机化
index_shuf = [i for i in range(X.shape[0])]
random.shuffle(index_shuf)
X = np.array([X[i] for i in index_shuf])
y = np.array([y[i] for i in index_shuf])
'''
#采样率 500
kernels, chans, samples = 1, 60, 851 #-0.2-1.5s
#kernels, chans, samples = 1, 60, 751 #0-1.5s
#kernels, chans, samples = 1, 60, 501 #0.2-1.2s
#kernels, chans, samples = 1, 60, 651 #0.2-1.5s
#kernels, chans, samples = 1, 60, 601 #0.-1.2s

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
print(X_test.shape[0], 'test samples') 

model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, 
               dropoutRate = 0.25, kernLength = 250, F1 = 4, D = 2, F2 = 8, 
               dropoutType = 'Dropout')

# load optimal weights
model.load_weights(CKP_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
acc = np.mean(preds == Y_test.argmax(axis=-1))
#print(probs, preds)
print("Classification accuracy: %f " % (acc))
#print('调用函数auc：', metrics.roc_auc_score(preds, Y_test, multi_class='ovr', average='macro'))
print('调用函数auc：', metrics.roc_auc_score(preds, Y_test, multi_class='ovr', average='weighted'))

chance_probs = np.full((571, ), 0.33)
#print(chance_probs,chance_probs.shape)
#print(probs.max(axis = -1), probs.max(axis = -1).shape)
stat, p = wilcoxon(probs.max(axis = -1) , chance_probs)
print('stat=%.4f, p=%.4f' % (stat, p))
if p > 0.05/571:
    print('Probably the same distribution')
else:
    print('Probably different distributions')

# plot the confusion matrices for both classifiers
names = ['JG', 'MM', 'JY']
#names = ['Negative', 'Neutral', 'Positive']
plt.figure(0)
#plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-4,2')
plt.show()

