import numpy as np
import random
# mne imports
import mne
from mne import io
from mne.datasets import sample

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
from sklearn.preprocessing import StandardScaler

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)

# while the default tensorflow ordering is 'channels_last' we set it here
# to be explicit in case if the user has changed the default ordering
K.set_image_data_format('channels_last')

'''-0.2-1.5s'''
#CKP_PATH = './tmp/checkpoint_intents_k5_82_17s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k12_82_17s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_82_17s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k64_82_17s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k5_42_17s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k5_41_17s_batch16.h5'
'''-0.2-1.2s'''
#CKP_PATH = './tmp/checkpoint_intents_k5_82_14s_batch64.h5'
#CKP_PATH = './tmp/checkpoint_intents_k12_82_14s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_82_14s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k64_82_14s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k5_42_14s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k5_41_14s_batch16.h5'
'''0-1.2s'''
#CKP_PATH = './tmp/checkpoint_intents_k5_82_12s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k12_82_12s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_82_12s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k64_82_12s_batch16.h5'

'''0-1.5s'''
#CKP_PATH = './new_save_models/checkpoint_intents_k25_81_15s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k25_81_15s_batch16_noscale.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k5_82_15s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k5_81_15s_batch16_scale1000.h5'

'''0.2-1.5s'''
#CKP_PATH = './new_save_models/checkpoint_intents_k64_82_13s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k25_82_13s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k12_82_13s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models_cross/checkpoint_intents_k12_82_13s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k5_82_13s_batch16_scale1000.h5'
#CKP_PATH = './new_save_models/checkpoint_intents_k64_81_13s_batch16_scale1000.h5'
CKP_PATH = './new_save_models/checkpoint_intents_k64_41_13s_batch16_scale1000.h5'

'''0.4-1.2s'''
#CKP_PATH = './tmp/checkpoint_intents_k5_82_8s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k12_82_8s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_82_8s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k64_82_8s_batch16.h5'

'''0.4-1.5s'''
#CKP_PATH = './tmp/checkpoint_intents_k5_82_11s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k12_82_11s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k25_82_11s_batch16.h5'
#CKP_PATH = './tmp/checkpoint_intents_k64_82_11s_batch16.h5'

#preprocessed_path = 'data/cleandata_sub24.mat'
#preprocessed_path = 'data'
preprocessed_path = 'enroll_data'
#feature_path = 'features'
#feature_path = 'features_scale1000_15s'
feature_path = 'features_scale1000'
#feature_path = 'features_scale1000_cross'

X_train      = np.load(feature_path+ '/' +'X_train.npy')
Y_train      = np.load(feature_path+ '/' +'Y_train.npy')
X_validate   = np.load(feature_path+ '/' +'X_validate.npy')
Y_validate   = np.load(feature_path+ '/' +'Y_validate.npy')
X_test       = np.load(feature_path+ '/' +'X_test.npy')
Y_test       = np.load(feature_path+ '/' +'Y_test.npy')
   
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
'''
#归一化-1
scaler = StandardScaler()
num_instances, num_features, num_time_steps = X_train.shape
X_train = np.reshape(X_train, newshape=(-1, num_features))
X_train = scaler.fit_transform(X_train)
X_train = np.reshape(X_train, newshape=(num_instances, num_features, num_time_steps))

num_instances, num_features, num_time_steps = X_test.shape
X_test = np.reshape(X_test, newshape=(-1, num_features))
X_test = scaler.transform(X_test)
X_test = np.reshape(X_test, newshape=(num_instances, num_features, num_time_steps))
X_validate = X_test
'''
'''
#归一化-2
scalers = {}
for i in range(X_train.shape[1]):
    scalers[i] = StandardScaler()
    X_train[:, i, :] = scalers[i].fit_transform(X_train[:, i, :]) 

for i in range(X_test.shape[1]):
    X_test[:, i, :] = scalers[i].transform(X_test[:, i, :]) 
X_validate = X_test
'''

############################# EEGNet portion ##################################
#采样率 500
#kernels, chans, samples = 1, 60, 851 #-0.2-1.5s
#kernels, chans, samples = 1, 60, 701 #-0.2-1.2s
#kernels, chans, samples = 1, 60, 601 #0-1.2s
#kernels, chans, samples = 1, 60, 751 #0-1.5s
kernels, chans, samples = 1, 60, 651 #0.2-1.5s
#kernels, chans, samples = 1, 60, 401 #0.4-1.2s
#kernels, chans, samples = 1, 60, 551 #0.4-1.5s
#kernels, chans, samples = 1, 60, 601 #-0.2-1.s

# convert data to NHWC (trials, channels, samples, kernels) format. Data 
# contains 60 channels and 151 time-points. Set the number of kernels to 1.
X_train      = X_train.reshape(X_train.shape[0], chans, samples, kernels)
X_validate   = X_validate.reshape(X_validate.shape[0], chans, samples, kernels)
X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)

# configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
# model configurations may do better, but this is a good starting point)
#model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, 
#               dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
#               dropoutType = 'Dropout')
model = EEGNet(nb_classes = 3, Chans = chans, Samples = samples, 
               dropoutRate = 0.0, kernLength = 64, F1 = 4, D = 1, F2 = 4, 
               dropoutType = 'Dropout')

# compile the model and set the optimizers
#model.compile(loss='categorical_crossentropy', optimizer='adam', 
#             metrics = ['accuracy'])
adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer=adam, 
             metrics = ['accuracy'])
#sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
#model.compile(loss='categorical_crossentropy', optimizer=sgd)

# count number of parameters in the model
#numParams  = model.count_params()  
#print("The number of paramas is ", numParams)  

# set a valid path for your system to record model checkpoints
checkpointer = ModelCheckpoint(filepath=CKP_PATH,verbose=1,
                               save_best_only=True)

###############################################################################
# if the classification task was imbalanced (significantly more trials in one
# class versus the others) you can assign a weight to each class during 
# optimization to balance it out. This data is approximately balanced so we 
# don't need to do this, but is shown here for illustration/completeness. 
###############################################################################

# the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
# the weights all to be 1
class_weights = {0:1, 1:1, 2:1}

################################################################################
# fit the model. Due to very small sample sizes this can get
# pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
# Riemannian geometry classification (below)
################################################################################
fittedModel = model.fit(X_train, Y_train, batch_size = 16, epochs = 200, 
                        verbose = 2, validation_data=(X_validate, Y_validate),
                        callbacks=[checkpointer], class_weight = class_weights)

###############################################    show   ###############################################

# 显示训练集和验证集的acc和loss曲线
#for key in fittedModel.history.keys():
#    print(key)
acc = fittedModel.history['accuracy']
val_acc = fittedModel.history['val_accuracy']
loss = fittedModel.history['loss']
val_loss = fittedModel.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# load optimal weights
model.load_weights(CKP_PATH)
# WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
# model.load_weights(WEIGHTS_PATH)

###############################################################################
# make prediction on test set.
###############################################################################

probs = model.predict(X_test)
preds = probs.argmax(axis = -1)  
acc = np.mean(preds == Y_test.argmax(axis=-1))
#print(probs, preds, Y_test.argmax(axis = -1))
print("Classification accuracy: %f " % (acc))

# plot the confusion matrices for both classifiers
names = ['JG', 'MM', 'JY']
plt.figure(0)
#plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,2')
#plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-8,1')
plot_confusion_matrix(preds, Y_test.argmax(axis = -1), names, title = 'EEGNet-4,1')
plt.show()
