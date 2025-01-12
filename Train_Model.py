import random
import torch
import torch.optim as optim
import numpy as np
import argparse
from torch_geometric.data import DataLoader
from torch.optim import lr_scheduler
from sklearn.metrics import accuracy_score
from process import cross_validation, CustomDataset
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import model_HGNN

EPS = 1e-10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--data_path', type=str, default="AD_HC", help='root directory of the dataset')
parser.add_argument('--batchSize', type=int, default=20, help='size of the batches')
parser.add_argument('--in_ch', type=int, default=187, help='feature dim')
parser.add_argument('--n_hid1', type=int, default=64, help='num of hiden')
parser.add_argument('--n_class', type=int, default=2, help='num of classes')
parser.add_argument('--drop_out', type=int, default=0.2, help='drop_out')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--gamma', type=float, default=0.5, help='scheduler shrinking rate')
parser.add_argument('--weight_decay', type=float, default=5e-4, help='regularization')
parser.add_argument('--num_epochs', type=int, default=500, help='starting epoch')
parser.add_argument('--seed', type=int, default=1, help='seed')
parser.add_argument('--derate_step', type=int, default=1, help='step_size')
parser.add_argument('--kfold', type=int, default=5, help='step_size')
opt = parser.parse_args()

torch.manual_seed(opt.seed)
np.random.seed(opt.seed)

random.seed(opt.seed)
np.random.seed(opt.seed)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
torch.cuda.manual_seed_all(opt.seed)


labels = torch.tensor(np.load("dataset/NC_AD_np_corresponding_graph_tags.npy"), dtype=torch.float32)
G = torch.tensor(np.load("dataset/AD_HC_G.npy"))
features = torch.tensor(np.load("dataset/NC_AD_np_fmri_as_feature.npy"))

shuffle_idx = np.random.permutation(len(G))
labels = labels[shuffle_idx]
G = G[shuffle_idx]
features = features[shuffle_idx]

train_index_list, test_index_list = cross_validation(labels, k_fold=opt.kfold)

acc_mean_fold,sensitivity_mean_fold,precision_mean_fold,f1_mean_fold,specificity_mean_fold = [], [], [], [], []
for fold_index in range(opt.kfold):
    model = model_HGNN.HGNNWrapper(in_ch=opt.in_ch,
                                   n_class=opt.n_class,
                                   n_hid1=opt.n_hid1,
                                   dropout=opt.drop_out).to(device)

    optimizer = optim.Adam(model.parameters(),
                           lr=opt.lr,
                           weight_decay=opt.weight_decay)

    scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.derate_step, gamma=opt.gamma, last_epoch=-1);
    criterion = torch.nn.CrossEntropyLoss()

    fold_train_index = train_index_list[fold_index]
    fold_test_index = test_index_list[fold_index]

    train_dataset = CustomDataset(G[fold_train_index], features[fold_train_index], labels[fold_train_index])
    test_dataset = CustomDataset(G[fold_test_index], features[fold_test_index], labels[fold_test_index])

    train_dataset = DataLoader(train_dataset, batch_size=opt.batchSize, shuffle=True)
    test_dataset = DataLoader(test_dataset, batch_size=opt.batchSize, shuffle=True)

    _pred, _label = [], []
    for epoch in range(opt.num_epochs):
        model.train()
        epoch_loss = 0
        for batch_idx, (adj_batch, features_batch, labels_batch) in enumerate(train_dataset):
            adj_batch, features_batch, labels_batch = adj_batch.numpy(), features_batch.numpy(), labels_batch.to(device)
            output = model(features_batch, adj_batch)

            labe = labels_batch.view(-1)
            loss = criterion(output, labe.long())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.detach().item()

            pred = torch.softmax(output, dim=1)
            pred = torch.max(pred, 1)[1].view(-1)
            _pred += pred.detach().cpu().numpy().tolist()
            _label += labels_batch.cpu().numpy().tolist()

        epoch_loss /= (batch_idx + 1)
        acc = accuracy_score(_label, _pred)
        print(('Epoch {}: train_loss {:.4f}, train_acc {:.4f}'.format(epoch+1, epoch_loss, acc)))



    model.eval()
    test_pred, test_label = [], []
    with ((torch.no_grad())):
        for batch_idx, (adj_batch, features_batch, labels_batch) in enumerate(test_dataset):
            adj_batch, features_batch, labels_batch = adj_batch.numpy(), features_batch.numpy(), labels_batch.to(device)
            pred = torch.softmax(model(features_batch, adj_batch), 1)
            pred = torch.max(pred, 1)[1].view(-1)
            test_pred += pred.detach().cpu().numpy().tolist()
            test_label += labels_batch.cpu().numpy().tolist()

        target = test_label
        pred = test_pred
        tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
        accuracy = accuracy_score(target, pred)
        sensitivity = recall_score(target, pred)
        specificity = tn / (tn + fp)
        precision = precision_score(target, pred)
        f1 = f1_score(target, pred)
        print('========================================== fold {}'.format(fold_index + 1),
              'result===============================================')
        print('Test_acc:{}, sensitivity:{}, specificity:{}, precision:{}, f1:{}'.format(accuracy, sensitivity,
                                                                                        specificity, precision, f1))
        print('=======================================================================================================')
        acc_mean_fold.append(accuracy)
        sensitivity_mean_fold.append(sensitivity)
        precision_mean_fold.append(precision)
        f1_mean_fold.append(f1)
        specificity_mean_fold.append(specificity)

print('===================================================== result ===================================================')
print('Test_acc:{}+{}, sensitivity:{}+{}, specificity:{}+{}, precision:{}+{}, f1:{}+{}'.format(
        np.mean(acc_mean_fold), np.std(acc_mean_fold), np.mean(sensitivity_mean_fold), np.std(sensitivity_mean_fold),
        np.mean(specificity_mean_fold), np.std(specificity_mean_fold),
        np.mean(precision_mean_fold), np.std(precision_mean_fold), np.mean(f1_mean_fold), np.std(f1_mean_fold)))
print('================================================================================================================')
