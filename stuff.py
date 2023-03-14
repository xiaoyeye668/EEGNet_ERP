import nibabel as nib
import numpy as np
import os
import math
from scipy.stats import spearmanr, pearsonr, kendalltau, ttest_1samp, ttest_rel
from skimage.measure import label
import sys
from collections import defaultdict

# get package abspath
package_root = os.path.dirname(os.path.abspath(__file__))

def permutation_test(v1, v2, iter=5000):

    """
    Conduct Permutation test

    Parameters
    ----------
    v1 : array
        Vector 1.
    v2 : array
        Vector 2.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
    """

    if len(v1) != len(v2):

        return "Invalid input"

    # permutation test

    diff = abs(np.average(v1) - np.average(v2))
    v = np.hstack((v1, v2))
    nv = v.shape[0]
    ni = 0

    for i in range(iter):
        vshuffle = np.random.permutation(v)
        vshuffle1 = vshuffle[:int(nv/2)]
        vshuffle2 = vshuffle[int(nv/2):]
        diff_i = np.average(vshuffle1) - np.average(vshuffle2)

        if diff_i >= diff:
            ni = ni + 1

    # permunitation test p-value
    p = np.float64(ni/iter)

    return p


' a function for permutation test for correlation coefficients '

def permutation_corr(v1, v2, method="spearman", iter=5000):

    """
    Conduct Permutation test for correlation coefficients

    Parameters
    ----------
    v1 : array
        Vector 1.
    v2 : array
        Vector 2.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
    """

    if len(v1) != len(v2):

        return "Invalid input"

    # permutation test

    if method == "spearman":

        rtest = spearmanr(v1, v2)[0]

        ni = 0

        for i in range(iter):
            v1shuffle = np.random.permutation(v1)
            v2shuffle = np.random.permutation(v2)
            rperm = spearmanr(v1shuffle, v2shuffle)[0]

            if rperm>rtest:
                ni = ni + 1

    if method == "pearson":
        print(iter)
        rtest = pearsonr(v1, v2)[0]

        ni = 0

        for i in range(iter):
            v1shuffle = np.random.permutation(v1)
            v2shuffle = np.random.permutation(v2)
            rperm = pearsonr(v1shuffle, v2shuffle)[0]

            if rperm>rtest:
                ni = ni + 1

    if method == "kendalltau":

        rtest = kendalltau(v1, v2)[0]

        ni = 0

        for i in range(iter):
            v1shuffle = np.random.permutation(v1)
            v2shuffle = np.random.permutation(v2)
            rperm = kendalltau(v1shuffle, v2shuffle)[0]

            if rperm>rtest:
                ni = ni + 1

    p = np.float64((ni+1)/(iter+1))

    return p


' a function for getting the 1-D & 1-sided cluster-index information '

def get_cluster_index_1d_1sided(m):

    """
    Get 1-D & 1-sided cluster-index information from a vector

    Parameters
    ----------
    m : array
        A significant vector.
        The values in m should be 0 or 1, which represent not significant point or significant point, respectively.

    Returns
    -------
    index_v : array
        The cluster-index vector.
    index_n : int
        The number of clusters.
    """

    x = np.shape(m)[0]
    b = np.zeros([x+2])
    b[1:x+1] = m

    index_v = np.zeros([x])

    index_n = 0
    for i in range(x):
        if b[i+1] == 1 and b[i] == 0 and b[i+2] == 1:
            index_n = index_n + 1
        if b[i+1] == 1:
            if b[i] != 0 or b[i+2] != 0:
                index_v[i] = index_n

    return index_v, index_n


' a function for getting the 1-D & 2-sided cluster-index information '

def get_cluster_index_1d_2sided(m):

    """
    Get 1-D & 2-sided cluster-index information from a vector

    Parameters
    ----------
    m : array
        A significant vector.
        The values in m should be 0 or 1 or -1, which represent not significant point or significantly higher point or
        significantly less point, respectively.

    Returns
    -------
    index_v1 : array
        The "greater" cluster-index vector.
    index_n1 : int
        The number of "greater" clusters.
    index_v2 : array
        The "less" cluster-index vector.
    index_n2 : int
        The number of "less" clusters.
    """

    x = np.shape(m)[0]
    b = np.zeros([x+2])
    b[1:x+1] = m

    index_v1 = np.zeros([x])

    index_n1 = 0
    for i in range(x):
        if b[i+1] == 1 and b[i] != 1 and b[i+2] == 1:
            index_n1 = index_n1 + 1
        if b[i+1] == 1:
            if b[i] == 1 or b[i+2] == 1:
                index_v1[i] = index_n1

    index_v2 = np.zeros([x])

    index_n2 = 0
    for i in range(x):
        if b[i + 1] == -1 and b[i] != -1 and b[i + 2] == -1:
            index_n2 = index_n2 + 1
        if b[i + 1] == -1:
            if b[i] == -1 or b[i + 2] == -1:
                index_v2[i] = index_n2

    return index_v1, index_n1, index_v2, index_n2


' a function for getting the 2-D & 1-sided cluster-index information '

def get_cluster_index_2d_1sided(m):

    """
    Get 2-D & 1-sided cluster-index information from a matrix

    Parameters
    ----------
    m : array
        A significant matrix.
        The values in m should be 0 or 1, which represent not significant point or significant point, respectively.

    Returns
    -------
    index_m : array
        The cluster-index matrix.
    index_n : int
        The number of clusters.
    """

    x, y = np.shape(m)
    b = np.zeros([x+2, y+2])
    b[1:x+1, 1:y+1] = m

    index_m = np.zeros([x, y])

    index_n = 0
    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            if b[ii, jj] == 1 and (b[ii-1, jj]+b[ii+1, jj]+b[ii, jj-1]+b[ii, jj+1]) != 0:
                min_index = index_n + 1
                if b[ii - 1, jj] == 1:
                    min_index = np.min([min_index, index_m[i - 1, j]])
                if b[ii, jj - 1] == 1:
                    min_index = np.min([min_index, index_m[i, j - 1]])
                k1 = 0
                while b[ii, jj - k1] == 1:
                    index_m[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b[ii - k2, jj] == 1:
                    index_m[i - k2, j] = min_index
                    k2 = k2 + 1
                k = 0
                while b[ii, jj + k] == 1:
                    index_m[i, j + k] = min_index
                    k = k + 1
                k = 0
                while b[ii + k, jj] == 1:
                    index_m[i + k, j] = min_index
                    k = k + 1
                if b[ii, jj - 1] != 1:
                    index_n = index_n + 1
                    k = 0
                    m = 0
                    while b[ii, jj + k] == 1:
                        if b[ii - 1, jj + k] == 1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n = index_n - 1

    return index_m, index_n


' a function for getting the 2-D & 2-sided cluster-index information '

def get_cluster_index_2d_2sided(m):

    """
    Get 2-D & 2-sided cluster-index information from a matrix

    Parameters
    ----------
    m : array
        A significant matrix.
        The values in m should be 0 or 1 or -1, which represent not significant point or significantly higher point or
        significantly less point, respectively.

    Returns
    -------
    index_m1 : array
        The "greater" cluster-index matrix.
    index_n1 : int
        The "greater" number of clusters.
    index_m2 : array
        The "less" cluster-index matrix.
    index_n2 : int
        The "less" number of clusters.
    """

    x, y = np.shape(m)
    b1 = np.zeros([x+2, y+2])
    b1[1:x+1, 1:y+1] = m
    b2 = np.zeros([x+2, y+2])
    b2[1:x+1, 1:y+1] = m
    index_m1 = np.zeros([x, y])
    index_m2 = np.zeros([x, y])
    index_n1 = 0
    index_n2 = 0

    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            index = True
            if b1[ii-1, jj] != 1 and b1[ii+1, jj] != 1 and b1[ii, jj-1] != 1 and b1[ii, jj+1] != 1:
                index = False
            if b1[ii, jj] == 1 and index == True:
                min_index = index_n1 + 1
                if b1[ii - 1, jj] == 1:
                    min_index = np.min([min_index, index_m1[i - 1, j]])
                if b1[ii, jj - 1] == 1:
                    min_index = np.min([min_index, index_m1[i, j - 1]])
                k1 = 0
                while b1[ii, jj - k1] == 1:
                    index_m1[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b1[ii - k2, jj] == 1:
                    index_m1[i - k2, j] = min_index
                    k2 = k2 + 1
                k3 = 0
                while b1[ii, jj + k3] == 1:
                    index_m1[i, j + k3] = min_index
                    k3 = k3 + 1
                k4 = 0
                while b1[ii + k4, jj] == 1:
                    index_m1[i + k4, j] = min_index
                    k4 = k4 + 1
                if b1[ii, jj - 1] != 1:
                    index_n1 = index_n1 + 1
                    k = 0
                    m = 0
                    while b1[ii, jj + k] == 1:
                        if b1[ii - 1, jj + k] == 1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n1 = index_n1 - 1

    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            index = True
            if b2[ii - 1, jj] != -1 and b2[ii + 1, jj] != -1 and b2[ii, jj - 1] != -1 and b2[ii, jj + 1] != -1:
                index = False
            if b2[ii, jj] == -1 and index == True:
                min_index = index_n2 + 1
                if b2[ii - 1, jj] == -1:
                    min_index = np.min([min_index, index_m2[i - 1, j]])
                if b2[ii, jj - 1] == -1:
                    min_index = np.min([min_index, index_m2[i, j - 1]])
                k1 = 0
                while b2[ii, jj - k1] == -1:
                    index_m2[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b2[ii - k2, jj] == -1:
                    index_m2[i - k2, j] = min_index
                    k2 = k2 + 1
                k = 0
                while b2[ii, jj + k] == -1:
                    index_m2[i, j + k] = min_index
                    k = k + 1
                k = 0
                while b2[ii + k, jj] == -1:
                    index_m2[i + k, j] = min_index
                    k = k + 1
                if b2[ii, jj - 1] != -1:
                    index_n2 = index_n2 + 1
                    k = 0
                    m = 0
                    while b2[ii, jj + k] == -1:
                        if b2[ii - 1, jj + k] == -1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n2 = index_n2 - 1

    return index_m1, index_n1, index_m2, index_n2


def get_cluster_index_2d_2sided_distance(m, distance=1):

    """
    Get 2-D & 2-sided cluster-index information from a matrix

    Parameters
    ----------
    m : array
        A significant matrix.
        The values in m should be 0 or 1 or -1, which represent not significant point or significantly higher point or
        significantly less point, respectively.
    distance : int
        neighbor distance , default=1
    """

    x, y = np.shape(m)
    b1 = np.zeros([x+2, y+2])
    b1[1:x+1, 1:y+1] = m
    b2 = np.zeros([x+2, y+2])
    b2[1:x+1, 1:y+1] = m
    index_m1 = np.zeros([x, y])
    index_m2 = np.zeros([x, y])
    index_n1 = 0
    index_n2 = 0

    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            index = True
            if b1[ii-1, jj] != 1 and b1[ii+1, jj] != 1 and b1[ii, jj-1] != 1 and b1[ii, jj+1] != 1:
                index = False
            if b1[ii, jj] == 1 and index == True:
                min_index = index_n1 + 1
                if b1[ii - 1, jj] == 1:
                    min_index = np.min([min_index, index_m1[i - 1, j]])
                if b1[ii, jj - 1] == 1:
                    min_index = np.min([min_index, index_m1[i, j - 1]])
                k1 = 0
                while b1[ii, jj - k1] == 1:
                    index_m1[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b1[ii - k2, jj] == 1:
                    index_m1[i - k2, j] = min_index
                    k2 = k2 + 1
                k = 0
                while b1[ii, jj + k] == 1:
                    index_m1[i, j + k] = min_index
                    k = k + 1
                print('<<<<<<<<<<< k1', k1,k)
                if abs(k - k1) <= distance :
                    continue

                k = 0
                while b1[ii + k, jj] == 1:
                    index_m1[i + k, j] = min_index
                    k = k + 1
                print('<<<<<<<<<<< k2', k2,k)
                if abs(k - k2) <= distance :
                    continue
                if b1[ii, jj - 1] != 1:
                    index_n1 = index_n1 + 1
                    k = 0
                    m = 0
                    while b1[ii, jj + k] == 1:
                        if b1[ii - 1, jj + k] == 1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n1 = index_n1 - 1
    for i in range(x):
        for j in range(y):
            ii = i + 1
            jj = j + 1
            index = True
            if b2[ii - 1, jj] != -1 and b2[ii + 1, jj] != -1 and b2[ii, jj - 1] != -1 and b2[ii, jj + 1] != -1:
                index = False
            if b2[ii, jj] == -1 and index == True:
                min_index = index_n2 + 1
                if b2[ii - 1, jj] == -1:
                    min_index = np.min([min_index, index_m2[i - 1, j]])
                if b2[ii, jj - 1] == -1:
                    min_index = np.min([min_index, index_m2[i, j - 1]])
                k1 = 0
                while b2[ii, jj - k1] == -1:
                    index_m2[i, j - k1] = min_index
                    k1 = k1 + 1
                k2 = 0
                while b2[ii - k2, jj] == -1:
                    index_m2[i - k2, j] = min_index
                    k2 = k2 + 1
                k = 0
                while b2[ii, jj + k] == -1:
                    index_m2[i, j + k] = min_index
                    k = k + 1
                k = 0
                while b2[ii + k, jj] == -1:
                    index_m2[i + k, j] = min_index
                    k = k + 1
                if b2[ii, jj - 1] != -1:
                    index_n2 = index_n2 + 1
                    k = 0
                    m = 0
                    while b2[ii, jj + k] == -1:
                        if b2[ii - 1, jj + k] == -1:
                            m = 1
                        k = k + 1
                    if m == 1:
                        index_n2 = index_n2 - 1
    return index_m1, index_n1, index_m2, index_n2

' a function for 1-sample & 1-sided cluster based permutation test for 1-D results '

def clusterbased_permutation_1d_1samp_1sided(results, level=0, p_threshold=0.05, iter=5000):

    """
    1-sample & 1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x]. n_subs represents the number of subjects.
    level : float. Default is 0.
        A expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1, which represent not significant point or significant
        point after cluster-based permutation test, respectively.
    """

    nsubs, x = np.shape(results)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = ttest_1samp(results[:, t], level, alternative='greater')
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        else:
            ps[t] = 0

    cluster_index, cluster_n = get_cluster_index_1d_1sided(ps)
    print(cluster_n)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t in range(x):
                if cluster_index[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t in range(x):
                    if cluster_index[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-p_threshold):
                for t in range(x):
                    if cluster_index[t] == i + 1:
                        ps[t] = 0

    return ps


' a function for 1-sample & 2-sided cluster based permutation test for 1-D results '

def clusterbased_permutation_1d_1samp_2sided(results, level=0, p_threshold=0.05, iter=5000):

    """
    1-sample & 2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x]. n_subs represents the number of subjects.
    level : float. Default is 0.
        A expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    ps : float
        The permutation test resultz, p-values.
        The shape of ps is [x]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test, respectively.
    """

    nsubs, x = np.shape(results)

    ps = np.zeros([x])
    ts = np.zeros([x])
    for t in range(x):
        ts[t], p = ttest_1samp(results[:, t], level, alternative='greater')
        if p < p_threshold and ts[t] > 0:
            ps[t] = 1
        ts[t], p = ttest_1samp(results[:, t], level, alternative='less')
        if p < p_threshold and ts[t] < 0:
            ps[t] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_1d_2sided(ps)
    print(cluster_n1, cluster_n2)

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t in range(x):
                if cluster_index1[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test\n")
        print("Side 1 begin:")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t in range(x):
                    if cluster_index1[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
                    
            if index < iter * (1-p_threshold):
                for t in range(x):
                    if cluster_index1[t] == i + 1:
                        ps[t] = 0

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t in range(x):
                if cluster_index2[t] == i + 1:
                    cluster_ts[i] = cluster_ts[i] + ts[t]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("Side 2 begin:\n")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t in range(x):
                    if cluster_index2[t] == j + 1:
                        v = np.hstack((results[:, t], chance))
                        vshuffle = np.random.permutation(v)
                        v1 = vshuffle[:nsubs]
                        v2 = vshuffle[nsubs:]
                        permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                print(cluster_ts[i],permu_ts[j])
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
                    
            if index < iter * (1-p_threshold):
                for t in range(x):
                    
                    if cluster_index2[t] == i + 1:
                        print(t,cluster_index2[t])
                        ps[t] = 0

    return ps


' a function for 1-sample & 1-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_1samp_1sided(results, level=0, p_threshold=0.05, iter=5000):

    """
    1-sample & 1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    level : float. Default is 0.
        A expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1, which represent not significant point or
        significant point after cluster-based permutation test, respectively.
    """

    nsubs, x1, x2 = np.shape(results)

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_1samp(results[:, t1, t2], level, alternative='greater')
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            else:
                ps[t1, t2] = 0

    cluster_index, cluster_n = get_cluster_index_2d_1sided(ps)
    print(cluster_n)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == j + 1:
                            v = np.hstack((results[:, t1, t2], chance))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-p_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    return ps


' a function for 1-sample & 2-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_1samp_2sided(results, level=0, p_threshold=0.05, iter=5000):

    """
    1-sample & 2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results : array
        A result matrix.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    level : float. Default is 0.
        A expected value in null hypothesis. (Here, results > level)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test, respectively.
    """

    nsubs, x1, x2 = np.shape(results)

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_1samp(results[:, t1, t2], level, alternative='greater')
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            ts[t1, t2], p = ttest_1samp(results[:, t1, t2], level, alternative='less')
            if p < p_threshold and ts[t1, t2] < 0:
                ps[t1, t2] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_2d_2sided(ps)
    print(cluster_n1, cluster_n2)

    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index1[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("\nPermutation test\n")
        print("Side 1 begin:")
        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == j + 1:
                            v = np.hstack((results[:, t1, t2], chance))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            #print('<<<<<<<<<<< permu_ts[i] ',permu_ts[i])
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1-p_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index2[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        chance = np.full([nsubs], level)
        print("Side 2 begin:\n")
        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == j + 1:
                            v = np.hstack((results[:, t1, t2], chance))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
            if index < iter * (1-p_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    return ps


' a function for 1-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_1sided(results1, results2, p_threshold=0.05, iter=5000):

    """
    1-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results1 : array
        A result matrix under condition1.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    results2 : array
        A result matrix under condition2.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects. (results1 > results2)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1, which represent not significant point or
        significant point after cluster-based permutation test.
    """

    nsubs1, x11, x12 = np.shape(results1)
    nsubs2, x21, x22 = np.shape(results2)

    if nsubs1 != nsubs2 and x11 != x21 and x12 != x22:

        return "Invalid input!"

    nsubs = nsubs1
    x1 = x11
    x2 = x12

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_rel(results1[:, t1, t2], results2[:, t1, t2], alternative="greater")
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            else:
                ps[t1, t2] = 0

    cluster_index, cluster_n = get_cluster_index_2d_1sided(ps)
    print(cluster_n)

    if cluster_n != 0:
        cluster_ts = np.zeros([cluster_n])
        for i in range(cluster_n):
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]

        permu_ts = np.zeros([iter])
        print("\nPermutation test")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n])
            for j in range(cluster_n):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == j + 1:
                            v = np.hstack((results1[:, t1, t2], results2[:, t1, t2]))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nCluster-based permutation test finished!\n")

        for i in range(cluster_n):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            if index < iter * (1 - p_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index[t1, t2] == i + 1:
                            ps[t1, t2] = 0

    return ps


' a function for 2-sided cluster based permutation test for 2-D results '

def clusterbased_permutation_2d_2sided(results1, results2, p_threshold=0.05, clusterp_threshold=0.05, threshold=6.0, iter=5000):

    """
    2-sided cluster based permutation test for 2-D results

    Parameters
    ----------
    results1 : array
        A result matrix under condition1.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects.
    results2 : array
        A result matrix under condition2.
        The shape of results should be [n_subs, x1, x2]. n_subs represents the number of subjects. (results1 > results2)
    p_threshold : float. Default is 0.05.
        The threshold of p-values.
    iter : int. Default is 5000.
        The times for iteration.

    Returns
    -------
    p : float
        The permutation test result, p-value.
        The shape of p is [x1, x2]. The values in ps should be 0 or 1 or -1, which represent not significant point or
        significantly greater point or significantly less point after cluster-based permutation test.
    """

    nsubs1, x11, x12 = np.shape(results1)
    nsubs2, x21, x22 = np.shape(results2)

    if nsubs1 != nsubs2 and x11 != x21 and x12 != x22:

        return "Invalid input!"

    nsubs = nsubs1
    x1 = x11
    x2 = x12

    ps = np.zeros([x1, x2])
    ts = np.zeros([x1, x2])
    for t1 in range(x1):
        for t2 in range(x2):
            ts[t1, t2], p = ttest_rel(results1[:, t1, t2], results2[:, t1, t2], alternative="greater")
            if p < p_threshold and ts[t1, t2] > 0:
                ps[t1, t2] = 1
            ts[t1, t2], p = ttest_rel(results1[:, t1, t2], results2[:, t1, t2], alternative="less")
            if p < p_threshold and ts[t1, t2] < 0:
                ps[t1, t2] = -1

    cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_2d_2sided(ps)
    #cluster_index1, cluster_n1, cluster_index2, cluster_n2 = get_cluster_index_2d_2sided_distance(ps, distance=3)
    print(cluster_n1, cluster_n2)

    count_time = defaultdict()
    if cluster_n1 != 0:
        cluster_ts = np.zeros([cluster_n1])
        for i in range(cluster_n1):
            count_time[i] = 0
            for t1 in range(x1):
                for t2 in range(x2):
                    if cluster_index1[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]
                        count_time[i] += 1
        print('<<<<<<< cluster_n1 ',i, count_time)
        permu_ts = np.zeros([iter])
        print("\nPermutation test\n")
        print("Side 1 begin:")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n1])
            for j in range(cluster_n1):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == j + 1:
                            v = np.hstack((results1[:, t1, t2], results2[:, t1, t2]))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            #cluster_t_values, cluster_p_values = ttest_rel(v1, v2, alternative="greater")
                            #if cluster_p_values < clusterp_threshold:
                            #   permu_cluster_ts[j] = permu_cluster_ts[j] + cluster_t_values
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="greater")[0]
            permu_ts[i] = np.max(permu_cluster_ts)
            #print('<<<<<<<<<<<< permu_ts[i]',permu_ts[i])
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 1 finished!\n")

        for i in range(cluster_n1):
            index = 0
            for j in range(iter):
                if cluster_ts[i] > permu_ts[j]:
                    index = index + 1
            #簇阈值控制
            print(cluster_ts[i], threshold)
            if abs(cluster_ts[i]) < threshold:
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == i + 1:
                            ps[t1, t2] = 0
            #if index < iter * (1 - p_threshold):
            if index < iter * (1 - clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index1[t1, t2] == i + 1:
                            #print('<<<<<<<<<<< cluster_index1[t1, t2] ', cluster_index1[t1, t2], i)
                            ps[t1, t2] = 0
    count_time = defaultdict()
    if cluster_n2 != 0:
        cluster_ts = np.zeros([cluster_n2])
        for i in range(cluster_n2):
            count_time[i] = 0
            for t1 in range(x1):               
                for t2 in range(x2):
                    if cluster_index2[t1, t2] == i + 1:
                        cluster_ts[i] = cluster_ts[i] + ts[t1, t2]
                        count_time[i] += 1
        print('<<<<<<< cluster_n2 ',i, count_time)
        permu_ts = np.zeros([iter])
        print("Side 2 begin:\n")

        for i in range(iter):
            permu_cluster_ts = np.zeros([cluster_n2])
            for j in range(cluster_n2):
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == j + 1:
                            v = np.hstack((results1[:, t1, t2], results2[:, t1, t2]))
                            vshuffle = np.random.permutation(v)
                            v1 = vshuffle[:nsubs]
                            v2 = vshuffle[nsubs:]
                            #cluster_t_values, cluster_p_values = ttest_rel(v1, v2, alternative="less")
                            #if cluster_p_values < clusterp_threshold:
                            #    permu_cluster_ts[j] = permu_cluster_ts[j] + cluster_t_values
                            permu_cluster_ts[j] = permu_cluster_ts[j] + ttest_rel(v1, v2, alternative="less")[0]
            #permu_ts[i] = np.max(permu_cluster_ts)
            permu_ts[i] = np.min(permu_cluster_ts)
            #print('<<<<<<<<<<<< permu_ts[i]',permu_ts[i])
            show_progressbar("Calculating", (i+1)*100/iter)
            if i == (iter - 1):
                print("\nSide 2 finished!\n")
                print("Cluster-based permutation test finished!\n")

        for i in range(cluster_n2):
            index = 0
            for j in range(iter):
                if cluster_ts[i] < permu_ts[j]:
                    index = index + 1
                    #print('<<<<<<<<<<<< index ', index, cluster_ts[i],permu_ts[j])
            #簇阈值控制
            print(cluster_ts[i], threshold)
            if abs(cluster_ts[i]) < threshold:
                for t1 in range(x1):
                    for t2 in range(x2):
                        if cluster_index2[t1, t2] == i + 1:
                            #print(abs(cluster_ts[i]),t1,t2,cluster_index2[t1, t2])
                            ps[t1, t2] = 0
            #if index < iter * (1 - p_threshold):
            if index < iter * (1 - clusterp_threshold):
                for t1 in range(x1):
                    for t2 in range(x2):
                        #print('<<<<<<<<<<< cluster_index2[t1, t2] ', cluster_index2[t1, t2], i)
                        if cluster_index2[t1, t2] == i + 1:
                            ps[t1, t2] = 0
    #删除孤立点
    for t1 in range(x1):
        for t2 in range(x2):
            if 0<t1<x1-1 and 0<t2<x2-1 and ps[t1, t2] != 0:
                if ps[t1-1, t2] ==0 and ps[t1+1, t2] ==0 and ps[t1, t2-1] ==0 and ps[t1, t2+1] ==0:
                    print(t1,t2)
                    ps[t1, t2] = 0
            elif t1==0 and ps[t1, t2] != 0:
                if ps[t1+1, t2] ==0 and  ps[t1, t2-1] ==0 and ps[t1, t2+1] ==0:
                    print(t1,t2)
                    ps[t1, t2] = 0
            elif t1==x1-1 and ps[t1, t2] != 0:
                if ps[t1-1, t2] ==0 and  ps[t1, t2-1] ==0 and ps[t1, t2+1] ==0:
                    print(t1,t2)
                    ps[t1, t2] = 0
            elif t2==0 and ps[t1, t2] != 0:
                if ps[t1-1, t2] ==0 and ps[t1+1, t2] ==0 and ps[t1, t2+1] ==0:
                    print(t1,t2)
                    ps[t1, t2] = 0
            elif t2==x2-1 and ps[t1, t2] != 0:
                if ps[t1-1, t2] ==0 and ps[t1+1, t2] ==0 and ps[t1, t2-1] ==0:
                    print(t1,t2)
                    ps[t1, t2] = 0

    return ps, cluster_n1, cluster_n2


' a function for showing the progress bar '

def show_progressbar(str, cur, total=100):

    percent = '{:.2%}'.format(cur / total)
    sys.stdout.write('\r')
    sys.stdout.write(str + ": [%-100s] %s" % ('=' * int(cur), percent))
    sys.stdout.flush()
