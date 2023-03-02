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
CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_17s_batch16_scale1000.h5'
feature_path = './datasets_cross_new/datasets_cross_17s_split85/'

X_test       = np.load(feature_path+ '/' +'X_test.npy')
Y_test       = np.load(feature_path+ '/' +'Y_test.npy')
   
print(X_test.shape[0], 'test samples')

#采样率 500
kernels, chans, samples = 1, 60, 851 #-0.2-1.5s
#kernels, chans, samples = 1, 60, 751 #0-1.5s
#kernels, chans, samples = 1, 60, 401 #0.4-1.2s
#kernels, chans, samples = 1, 60, 651 #0.2-1.5s
#kernels, chans, samples = 1, 60, 601 #-0.2-1.s

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
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
acc = np.mean(preds == Y_test.argmax(axis=-1))
print(probs.shape)
print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
names = ['JG', 'MM', 'JY']
plt.figure(0)
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
#plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-4,2')
plt.show()

#print('调用函数auc：', metrics.roc_auc_score(preds, Y_test, multi_class='ovr', average='macro'))
print('调用函数auc：', metrics.roc_auc_score(preds, Y_test, multi_class='ovr', average='weighted'))
chance_probs = np.full((394, ), 0.33)
#print(chance_probs,chance_probs.shape)
#print(probs.max(axis = -1), probs.max(axis = -1).shape)
stat, p = wilcoxon(probs.max(axis = -1) , chance_probs)
print('stat=%.4f, p=%.4f' % (stat, p))
if p > 0.05:
    print('Probably the same distribution')
else:
    print('Probably different distributions')
