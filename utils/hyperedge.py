import numpy as np

def construct_H_with_diff(diff_mat):
    num_rows, num_columns = diff_mat.shape
    absolute_diff = diff_mat.detach().numpy()
    H_List = []
    for i in range(num_rows):
        threshold = np.median(absolute_diff[i])
        condition = absolute_diff[i] > threshold
        H_List.append(list(np.where(condition)[0]))
    return H_List
def hyperedge_concat(H_list):
    non_empty_h = [h for h in H_list]# if len(h) > 1]
    H = np.vstack([np.array([1.0 if i in h else 0.0 for i in range(len(H_list))]) for h in non_empty_h])
    return H.T if H.size > 0 else H

def generate_G_from_H(H, variable_weight=False):
    if type(H) != list:
        return _generate_G_from_H(H, variable_weight)
    else:
        G = []
        for sub_H in H:
            G.append(generate_G_from_H(sub_H, variable_weight))
        return G

def symmetric_normalize(matrix):
    degree_matrix = np.diag(np.sum(matrix, axis=0))
    degree_matrix_inv_sqrt = np.linalg.inv(np.sqrt(degree_matrix))
    normalized_matrix = np.dot(np.dot(degree_matrix_inv_sqrt, matrix), degree_matrix_inv_sqrt)
    return normalized_matrix

def _generate_G_from_H(H, variable_weight=False):# 从临界矩阵到图
    H = np.array(H)
    H = symmetric_normalize(H)
    n_edge = H.shape[1]
    W = np.ones(n_edge)
    DV = np.sum(H * W, axis=1)
    DE = np.sum(H, axis=0)
    invDE = np.mat(np.diag(np.power(DE, -1))) # pow(De,-1)
    DV2 = np.mat(np.diag(np.power(DV, -0.5))) # pow(Dv,-1)
    W = np.mat(np.diag(W)) # 权重矩阵
    H = np.mat(H) # 邻接矩阵
    HT = H.T

    if variable_weight:
        DV2_H = DV2 * H
        invDE_HT_DV2 = invDE * HT * DV2

        return DV2_H, W, invDE_HT_DV2
    else:
        G = DV2 * H * W * invDE * HT * DV2
        return G # 返回的其实是图的拉普拉斯矩阵