import os
import numpy as np
import torch
import utils.graph_learn as hy
from scipy.signal import hilbert

def load_G(x):
    x = z_score_standardization(x)
    x = torch.tensor(x).float()
    plv = compute_PLV_matrix(x.T)
    plv = torch.Tensor(plv)
    G = hy.G(x, plv)
    return G

def compute_PLV_matrix(ROISignals):
    N, V = ROISignals.shape
    hilbert_mat = np.empty((N, V))
    for i in range(N):
        analytic_signal = hilbert(ROISignals[i])  # 希尔伯特变换
        amplitude_envelope = np.abs(analytic_signal)  # 获取振幅
        instantaneous_phase = np.angle(analytic_signal)  # 获取相位
        hilbert_mat[i, :] = instantaneous_phase

    N, _ = hilbert_mat.shape
    PLV_matrix = np.empty((N, N))
    for i in range(N):
        for j in range(N):
            phase_diff = hilbert_mat[i] - hilbert_mat[j]
            plv = np.abs(np.mean(np.exp(1j * phase_diff)))
            PLV_matrix[i, j] = plv
    return PLV_matrix

def z_score_standardization(matrix):
    mean = np.mean(matrix)
    std = np.std(matrix)
    return (matrix - mean) / std

if __name__ == '__main__':
    data = np.load("dataset/NC_AD_np_fmri_as_feature.npy")
    G_list = []
    for i in range(len(data)):
        G = load_G(data[i])
        G_list.append(G)
    matrix_array = np.array(G_list)
    np.save("dataset/AD_HC_G.npy", matrix_array)


