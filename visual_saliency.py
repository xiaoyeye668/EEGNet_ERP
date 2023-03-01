
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
#print(probs[150:200], Y_test.argmax(axis=-1)[150:200])
acc = np.mean(preds == Y_test.argmax(axis=-1))
print('调用函数auc：', metrics.roc_auc_score(preds, Y_test, multi_class='ovr', average='weighted'))

'''
Saliency Map
'''
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils import normalize
#Implement functions required to use attentions
#model modifier
def model_modifier_function(cloned_model):
    cloned_model.layers[-2].activation = tf.keras.activations.linear
replace2linear = model_modifier_function(model)

#score function
from tf_keras_vis.utils.scores import CategoricalScore
score = CategoricalScore([0, 1, 2])

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)

# Generate saliency map with smoothing that reduce noise by adding noise
# 0-221 1-3 2-203
#test_1, test_2, test_3 = X_test[0], X_test[2], X_test[6] 2103
test_1, test_2, test_3 = X_train[221], X_train[3], X_train[203]

X = np.asarray([test_1, test_2, test_3])

print('<<<<<<<< X ', X.shape,Y_train[221], Y_train[3], Y_train[203])
probs = model.predict(X)
preds = probs.argmax(axis = -1) 
probs = probs.max(axis = -1)   

saliency_map = saliency(score,
                        X,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20) # noise spread level.

# Render
#plt.title=('Saliency Map')

y = list(range(0,60))
ch_names = np.loadtxt('Xiaoqing60_AF7.txt', dtype=str, usecols=-1)
ch_names = list(ch_names)
ch_names.remove('COMNT')
ch_names.remove('SCALE')
ch_names.reverse()

x=[0,250,500,650]
#x=[200,250,500,750]
values = [i/500+0.2 for i in x]

for i, title in enumerate(test_titles):
    #plt.subplot(1, 3, i+1)
    plt.figure(figsize=(10, 8))
    #plt.title(title+' {}'.format(probs[i]), fontsize=12)
    plt.title(title+' {}'.format('salency map'), fontsize=12)
    plt.xticks(x, values)
    plt.yticks(y, ch_names)
    print(saliency_map[i].shape)
    #100:600  0.4s-1.4s
    #100:500  0.4s-1.2s
    #img = plt.imshow(saliency_map[i][:,100:500], cmap='jet')   
    img = plt.imshow(saliency_map[i], cmap='jet', aspect='auto') 
    #img = plt.imshow(saliency_map[i])   
    #ax[i].yaxis.set_tick_params(which='major',length=4,labelsize=12)
    #ax[i].axis('off')
    plt.colorbar(img)
    plt.show()
