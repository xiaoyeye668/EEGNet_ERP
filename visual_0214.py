
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.pyplot import MultipleLocator
# EEGNet-specific imports
from EEGModels import EEGNet
import tensorflow as tf
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import backend as K

from tf_keras_vis.utils import num_of_gpus
_, gpus = num_of_gpus()
print('Tensorflow recognized {} GPUs'.format(gpus))

SEED=1234
tf.random.set_seed(SEED)
np.random.seed(SEED)
K.set_image_data_format('channels_last')

CKP_PATH = './save_models_cross/checkpoint_intents_k64_82_13s_batch16_scale1000.h5'

#Load test dataset
test_titles = ['JG', 'MM', 'JY']
feature_path = './datasets_cross/datasets_cross_13s/'

X_test       = np.load(feature_path+ '/' +'X_test.npy')
Y_test       = np.load(feature_path+ '/' +'Y_test.npy')

#kernels, chans, samples = 1, 60, 751 #0-1.5s
kernels, chans, samples = 1, 60, 651 #0.2-1.5s

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
'''
Saliency Map
'''
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
# from tf_keras_vis.utils import normalize
#Implement functions required to use attentions
#model modifier
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
#replace2linear = ReplaceToLinear()
# Instead of using the ReplaceToLinear instance above,
# you can also define the function from scratch as follows:
def model_modifier_function(cloned_model):
    #cloned_model.layers[-1].activation = tf.keras.activations.linear
    cloned_model.layers[-2].activation = tf.keras.activations.linear
replace2linear = model_modifier_function(model)

#score function
from tf_keras_vis.utils.scores import CategoricalScore
# 1 is the imagenet index corresponding to Goldfish, 294 to Bear and 413 to Assault Rifle.
#score = CategoricalScore([1, 294, 413])
score = CategoricalScore([0, 1, 2])

# Instead of using CategoricalScore object,
# you can also define the function from scratch as follows:
def score_function(output):
    # The `output` variable refers to the output of the model,
    # so, in this case, `output` shape is `(3, 1000)` i.e., (samples, classes).
    #return (output[0][1], output[1][294], output[2][413])
    return (output[0][0], output[132][1], output[6][2])
print('<<<<<<< output ', model.layers[-2].output, model.layers[-2].output.shape)
#score = score_function(output = model.layers[-2].output)

# Create Saliency object.
saliency = Saliency(model,
                    model_modifier=replace2linear,
                    clone=True)

# Generate saliency map with smoothing that reduce noise by adding noise
#print(X_test[0].shape)
#X_test       = X[0]     #0 -- 2; 2 134 132 -- 3; 6 -- 4
#Y_test       = y[0]

test_1, test_2, test_3 = X_test[0], X_test[2], X_test[6]
X = np.asarray([test_1, test_2, test_3])

print('<<<<<<<< X ', X.shape)

#X_test       = X_test.reshape(X_test.shape[0], chans, samples, kernels)
saliency_map = saliency(score,
                        X,
                        smooth_samples=20, # The number of calculating gradients iterations.
                        smooth_noise=0.20) # noise spread level.

## Since v0.6.0, calling `normalize()` is NOT necessary.
# saliency_map = normalize(saliency_map)

# Render
#plt.title=('Saliency Map')

y = list(range(0,60))
#y = [i/60 for i in range(0,60)]
ch_names = ['FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ','F2','F4','F6',
'F8','FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ',
'C2','C4','C6','T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5',
'P3','P1','PZ','P2','P4','P6','P8','PO7','PO5','PO3','POZ','PO4','PO6','PO8','O1','OZ','O2']
print(len(y),len(ch_names))

x=[0,250,500,650]
#x=[200,250,500,750]
values = [i/500+0.2 for i in x]
'''
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10,4),sharey=True,sharex=True)
for i, title in enumerate(test_titles):
    ax[i].set_title(title, fontsize=8)
    ax[i].set_yticks(y, ch_names)
    ax[i].set_xticks(x, values)
    img = ax[i].imshow(saliency_map[i], cmap='jet')   
    #ax[i].yaxis.set_tick_params(which='major',length=4,labelsize=12)
    #ax[i].axis('off')
f.colorbar(img)
plt.tight_layout()
#plt.savefig('images/smoothgrad.png')
plt.show()
'''
for i, title in enumerate(test_titles):
    #plt.subplot(1, 3, i+1)
    plt.figure(figsize=(10, 8))
    plt.title(title+' Saliency Map', fontsize=12)
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
#f.colorbar(img)
#plt.show()

'''
from matplotlib import cm
from tf_keras_vis.gradcam_plus_plus import GradcamPlusPlus

# Create GradCAM++ object
gradcam = GradcamPlusPlus(model,
                          model_modifier=replace2linear,
                          clone=True)

# Generate heatmap with GradCAM++
cam = gradcam(score,
              X,
              penultimate_layer=-1)

## Since v0.6.0, calling `normalize()` is NOT necessary.
# cam = normalize(cam)

# Render
plt.title=('GradCAM++')
f, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 4))
for i, title in enumerate(test_titles):
    #heatmap = np.uint8(cm.jet(cam[i])[..., :3] * 255)
    heatmap = np.uint8(cm.jet(cam[i])[..., :3])
    ax[i].set_title(title, fontsize=8)
    ax[i].set_xticks(x, values)
    ax[i].imshow(X[i])
    img = ax[i].imshow(heatmap, cmap='jet', alpha=0.5)
    #ax[i].axis('off')
plt.tight_layout()
f.colorbar(img)
plt.show()
'''