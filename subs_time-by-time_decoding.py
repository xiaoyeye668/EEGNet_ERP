# -*- coding: utf-8
import scipy.io as sio
import numpy as np

def read_from_mat(file_path):
    raw_data = sio.loadmat(file_path)
    data = raw_data['data_2visual']
    chas_arr = data['label'][0][0]
    fsample_arr = data['fsample'][0][0]
    features_arr = data['trial'][0][0].T
    label_arr = data['trialinfo'][0][0]
    return chas_arr,fsample_arr,features_arr,label_arr

preprocessed_path = 'enroll_data'
# 被试id
subs = ["sub01", "sub02", "sub03", "sub05", "sub06", 'sub07','sub10','sub11',
'sub12','sub13', 'sub14', 'sub15', 'sub16', 'sub17', 'sub18', 'sub19','sub20','sub21',
'sub22','sub23','sub24','sub25','sub26','sub27','sub28','sub29','sub30','sub31']

data = np.zeros([28, 128, 60, 851])
label = np.zeros([28, 128])

sub_index = 0
for sub in subs:
    temp_features = []
    temp_labels = []
    data_path = preprocessed_path + '/cleandata_' + sub + '.mat'
    # 加载单个被试的ERP数据
    chas_arr,fsample_arr, features_arr,label_arr = read_from_mat(data_path)
    
    # shape of features: 139*60*851
    # 139 - 试次数； 60 - 导联； 851 - 时间点  500Hz， 1.7s [-0.2s至1.5s]
    assert features_arr.shape[0] == label_arr.shape[0]
    n_channels, fs = chas_arr.shape[0], fsample_arr[0][0]
    print(features_arr.shape[0])
    #nSamples = features_arr.shape[0]
    nSamples = 128
    n_timepoints = int(fs*1.7+1)
    print(nSamples,n_timepoints)
    for i in range(nSamples):
        temp_features.append(np.zeros([n_channels, n_timepoints]))
    temp_labels = np.zeros([nSamples])

    for i in range(nSamples):
        temp_features[i][:,:] = features_arr[i][0]
        temp_labels[i] = label_arr[i][0]

    subdata = np.array(temp_features)
    sublabel = np.array(temp_labels)

    data[sub_index] = subdata
    label[sub_index] = sublabel

    sub_index = sub_index + 1

from neurora.decoding import tbyt_decoding_kfold
print(data.shape)
accs = tbyt_decoding_kfold(data, label, n=3, navg=5, time_win=5, time_step=5, nfolds=3, nrepeats=50, smooth=True)
print(accs.shape)
np.savetxt("results_all.txt", accs)