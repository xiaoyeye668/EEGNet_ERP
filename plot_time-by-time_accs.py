# -*- coding: utf-8
from neurora.rsa_plot import plot_tbyt_decoding_acc
import numpy as np
from matplotlib import pyplot as plt

accs = np.loadtxt("results_all.txt")
print(accs.shape)
plot_tbyt_decoding_acc(accs, start_time=-0.2, end_time=1.5, time_interval=0.01, chance=0.33, p=0.01, cbpt=False,
                           stats_time=[0.0, 1.5], color='r', xlim=[0, 1.5], ylim=[0.2, 1.0], figsize=[6.4, 3.6], x0=0,
                           fontsize=8, avgshow=False)
