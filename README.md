# EEGNet_ERP
基于EEGNet的ERP解码分析
代码运行环境:Python>=3.7
主要用到的工具包:numpy, tensorflow, mne, scikit-learn, neurora

1. 数据集处理 -- ERP_datasets.py
2. EEGNet模型构建 -- EEGModel.py
3. 模型训练 -- ERP_3class_subs_showtrain_new.py
4. 模型性能 -- test_subs_new.py
5. 输入信号与分类决策的saliency map可视化 -- visual_saliency.py
6. spatial filter可视化 && 时频分析 -- visual_spatial.py

#SVM 
1. 时序解码器训练 -- subs_time-by-time_decoding.py
2. 解码结果可视化 -- plot_time-by-time_accs.py


