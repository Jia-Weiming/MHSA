from sklearn.model_selection import StratifiedKFold
import numpy as np
from torch.utils.data import Dataset

def cross_validation(label, k_fold):
    skf = StratifiedKFold(n_splits=k_fold, shuffle=True, random_state=1)
    train_index_list, test_index_list = [], []
    for train_index, test_index in skf.split(np.zeros(len(label)), label):
        train_index_list.append(train_index)
        test_index_list.append(test_index)

    return train_index_list, test_index_list

class CustomDataset(Dataset):
    def __init__(self, adj, features, labels):
        self.adj = adj
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        return self.adj[index], self.features[index], self.labels[index]
