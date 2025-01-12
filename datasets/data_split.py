from sklearn.model_selection import KFold
import random

def train_test_split_single(n_sub,kfold=5, fold=1):
    id = list(range(n_sub))
    #random.seed(123)
    #random.shuffle(id)
    kf = KFold(n_splits=kfold, random_state=3407, shuffle=True)

    # 选择第 fold 折作为测试集，其余作为训练集
    for i, (train_index, test_index) in enumerate(kf.split(id)):
        if i == fold:
            return train_index, test_index