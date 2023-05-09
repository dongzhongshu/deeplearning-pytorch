import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import matplotlib
import requests

# %matplotlib inline
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
from d2l.torch import plot

train_data = pd.read_csv('./data/kaggle_house_pred_train.csv')
# train_data = train_data.iloc[0:100, :]
test_data = pd.read_csv('./data/kaggle_house_pred_test.csv')
# test_data = test_data.iloc[0:100, :]
from sklearn.model_selection import StratifiedKFold



print(train_data.shape)
print(test_data.shape)


all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
print(all_features)


all_features["MSZoning"] = all_features["MSZoning"].str.replace("<", "NA").astype(str)
MSZonings = sorted(set(all_features.loc[:, 'MSZoning'].str.lower()))
MSZonings_value_2_i = dict([(v, i) for i, v in enumerate(MSZonings)])
all_features["MSZoning"] = all_features["MSZoning"].str.lower().map(MSZonings_value_2_i)
print(all_features["MSZoning"])

Streets = sorted(set(all_features.loc[:, 'Street'].str.lower()))
Streets_value_2_i = dict([(v, i) for i, v in enumerate(Streets)])
all_features["Street"] = all_features["Street"].str.lower().map(Streets_value_2_i)
print(all_features["Street"])

all_features["LotFrontage"].fillna(all_features["LotFrontage"].mean(), inplace=True)
#
all_features["Alley"] = all_features["Alley"].str.replace("<", "NA").astype(str)  # 一定要用astype
Alleys = sorted(set(all_features["Alley"]))
Alleys_value_2_i = dict([(v, i) for i, v in enumerate(Alleys)])
all_features["Alley"] = all_features["Alley"].str.lower().map(Alleys_value_2_i)
print(all_features["Alley"])

LotShapes = sorted(set(all_features["LotShape"].str.lower()))
LotShapes_value_2_i = dict([(v, i) for i, v in enumerate(LotShapes)])
all_features["LotShape"] = all_features["LotShape"].str.lower().map(LotShapes_value_2_i)
print("------------------")
print(all_features["LotShape"])


LandContours = sorted(set(all_features["LandContour"].str.lower()))
LandContours_value_2_i = dict([(v, i) for i, v in enumerate(LandContours)])
all_features["LandContour"] = all_features["LandContour"].str.lower().map(LandContours_value_2_i)
print("------------------")
print(all_features["LandContour"])


all_features["Utilities"] = all_features["Utilities"].str.replace("<", "NA").astype(str)
Utilities = sorted(set(all_features["Utilities"].str.lower()))
Utilities_value_2_i = dict([(v, i) for i, v in enumerate(Utilities)])
all_features["Utilities"] = all_features["Utilities"].str.lower().map(Utilities_value_2_i)
print("------------------")
print(all_features["Utilities"])


LotConfigs = sorted(set(all_features["LotConfig"].str.lower()))
LotConfigs_value_2_i = dict([(v, i) for i, v in enumerate(LotConfigs)])
all_features["LotConfig"] = all_features["LotConfig"].str.lower().map(LotConfigs_value_2_i)
print("------------------")
print(all_features["LotConfig"])


LandSlopes = sorted(set(all_features["LandSlope"].str.lower()))
LandSlopes_value_2_i = dict([(v, i) for i, v in enumerate(LandSlopes)])
all_features["LandSlope"] = all_features["LandSlope"].str.lower().map(LotConfigs_value_2_i)
print("------------------")
print(all_features["LandSlope"])


Neighborhoods = sorted(set(all_features["Neighborhood"].str.lower()))
Neighborhoods_value_2_i = dict([(v, i) for i, v in enumerate(Neighborhoods)])
all_features["Neighborhood"] = all_features["Neighborhood"].str.lower().map(Neighborhoods_value_2_i)
print("------------------")
print(all_features["Neighborhood"])





Condition1s = sorted(set(all_features["Condition1"].str.lower()))
Condition1s_value_2_i = dict([(v, i) for i, v in enumerate(Condition1s)])
all_features["Condition1"] = all_features["Condition1"].str.lower().map(Condition1s_value_2_i)
print("------------------")
print(all_features["Condition1"])

Condition2s = sorted(set(all_features["Condition2"].str.lower()))
Condition2s_value_2_i = dict([(v, i) for i, v in enumerate(Condition2s)])
all_features["Condition2"] = all_features["Condition2"].str.lower().map(Condition2s_value_2_i)
print("------------------")
print(all_features["Condition2"])

BldgTypes = sorted(set(all_features["BldgType"].str.lower()))
BldgTypes_value_2_i = dict([(v, i) for i, v in enumerate(BldgTypes)])
all_features["BldgType"] = all_features["BldgType"].str.lower().map(BldgTypes_value_2_i)
print("------------------")
print(all_features["BldgType"])


HouseStyles = sorted(set(all_features["HouseStyle"].str.lower()))
HouseStyles_value_2_i = dict([(v, i) for i, v in enumerate(HouseStyles)])
all_features["HouseStyle"] = all_features["HouseStyle"].str.lower().map(HouseStyles_value_2_i)
print("------------------")
print(all_features["HouseStyle"])


RoofStyles = sorted(set(all_features["RoofStyle"].str.lower()))
RoofStyles_value_2_i = dict([(v, i) for i, v in enumerate(RoofStyles)])
all_features["RoofStyle"] = all_features["RoofStyle"].str.lower().map(RoofStyles_value_2_i)
print("------------------")
print(all_features["RoofStyle"])



RoofMatls = sorted(set(all_features["RoofMatl"].str.lower()))
RoofMatls_value_2_i = dict([(v, i) for i, v in enumerate(RoofMatls)])
all_features["RoofMatl"] = all_features["RoofMatl"].str.lower().map(RoofMatls_value_2_i)
print("------------------")
print(all_features["RoofMatl"])



all_features["Exterior1st"] = all_features["Exterior1st"].str.replace("<", "NA").astype(str)
Exterior1sts = sorted(set(all_features["Exterior1st"].str.lower()))
Exterior1sts_value_2_i = dict([(v, i) for i, v in enumerate(Exterior1sts)])
all_features["Exterior1st"] = all_features["Exterior1st"].str.lower().map(Exterior1sts_value_2_i)
print("------------------")
print(all_features["Exterior1st"])


all_features["Exterior2nd"] = all_features["Exterior2nd"].str.replace("<", "NA").astype(str)
Exterior2nds = sorted(set(all_features["Exterior2nd"].str.lower()))
Exterior2nds_value_2_i = dict([(v, i) for i, v in enumerate(Exterior2nds)])
all_features["Exterior2nd"] = all_features["Exterior2nd"].str.lower().map(Exterior2nds_value_2_i)
print("------------------")
print(all_features["Exterior2nd"])




# all_features = train_data.iloc[:, 1:-1]
# numeric_features = all_features.dtypes[all_features.dtypes != 'str'].index
# numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# print(numeric_features[:])

# all_features = pd.get_dummies(all_features, dummy_na=True)


all_features = all_features[all_features.dtypes[all_features.dtypes != 'str'].index]
all_features["LandSlope"].fillna(0, inplace=True)
all_features = all_features[all_features.dtypes[all_features.dtypes != 'object'].index]
indexs = all_features.dtypes[all_features.dtypes != 'float64'].index
# all_features = all_features[:].apply(
#     lambda x: (x - x.mean()) / (x.std()))
# # 在标准化数据之后，所有均值消失，因此我们可以将缺失值设置为0
all_features[:] = all_features[:].fillna(0)
n_train = train_data.shape[0]
print(n_train)
train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32)
# for i in range(331):
#     print(train_features[0][i])
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32)
train_labels = torch.tensor(
    train_data.SalePrice.values.reshape(-1, 1), dtype=torch.float32)

loss = nn.MSELoss()
in_features = train_features.shape[1]


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return len(self.data)


def get_net():
    print(in_features)
    # net = nn.Sequential(nn.Linear(in_features,1))
    # net = nn.Sequential(nn.Flatten(), nn.Linear(in_features, 16), nn.ReLU(),nn.Dropout(0.1),  nn.Linear(16, 32), nn.ReLU(),
    #                  nn.Linear(32, 1))

    #torch.nn.BatchNorm1d(in_features),
    net = nn.Sequential( nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, 1))
    # net = nn.Sequential(nn.Linear(in_features, 1))
    # net = nn.Sequential(nn.Linear(in_features, 128), nn.ReLU(), nn.Linear(128, 1))
    return net


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def loss_batch(model, loss_func, xb, yb, opt=None):
    z = model(xb)
    clipped_preds = z.clamp(1, float('inf'))
    lossval = loss_func(torch.log(z), torch.log(yb))
    if opt is not None:
        lossval.backward()
        opt.step()
        opt.zero_grad()
    return lossval.item(), len(xb)

def train(net, train_features, train_labels, test_features, test_labels, num_epochs, learning_rate, weight_decay,
          batch_size):
    train_ls, test_ls = [], []
    datasets = MyDataset(train_features, train_labels)
    train_iter = torch.utils.data.DataLoader(dataset=datasets, batch_size=batch_size, shuffle=True)
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate, weight_decay=weight_decay)
    validset = MyDataset(test_features, test_labels)
    test_loader = torch.utils.data.DataLoader(dataset=validset, batch_size=batch_size, shuffle=True)
    for epoch in range(num_epochs):
        net.train()
        for X, y in train_iter:
            optimizer.zero_grad()
            for i in X:
                if torch.isnan(i).any():
                    print(i)

            clipped_preds = net(X)
            clipped_preds = clipped_preds.clamp(1, float('inf'))
            l = loss(torch.log(clipped_preds), torch.log(y))
            with torch.autograd.detect_anomaly():
                l.backward()
            optimizer.step()
        net.eval()
        if test_labels is not None:
            with torch.no_grad():
                data = [loss_batch(net, loss, x, y) for x, y in test_loader]
                loss_items, nums = zip(*data)
            val_loss = np.sum(np.multiply(loss_items, nums)) / np.sum(nums)
            test_ls.append(val_loss)
    return train_ls, test_ls


# def get_k_fold_data(k, i, X, y):
#     assert k > 1
#     fold_size = X.shape[0] // k
#     X_train, y_train = None, None
#     for j in range(k):
#         idx = slice(j * fold_size, (j + 1) * fold_size)
#         X_part, y_part = X[idx, :], y[idx]
#         if j == i:
#             X_valid, y_valid = X_part, y_part
#         elif X_train is None:
#             X_train, y_train = X_part, y_part
#         else:
#             X_train = torch.cat([X_train, X_part], 0)
#             y_train = torch.cat([y_train, y_part], 0)
#     return X_train, y_train, X_valid, y_valid
#
#
# def k_fold(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
#            batch_size):
#     train_l_sum, valid_l_sum = 0, 0
#     for i in range(k):
#         data = get_k_fold_data(k, i, X_train, y_train)
#         net = get_net()
#         train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
#                                    weight_decay, batch_size)
#         train_l_sum += train_ls[-1]
#         valid_l_sum += valid_ls[-1]
#         if i == 0:
#             d2l.plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
#                      xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
#                      legend=['train', 'valid'], yscale='log')
#         print(f'折{i + 1}，训练log rmse{float(train_ls[-1]):f}, '
#               f'验证log rmse{float(valid_ls[-1]):f}')
#     return train_l_sum / k, valid_l_sum / k

# k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 32
def fit(k, X_train, y_train, num_epochs, learning_rate, weight_decay,
        batch_size):
    train_l_sum, valid_l_sum = 0, 0
    skf = StratifiedKFold(n_splits=k)
    net = get_net()
    for i, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train)):
        trainset, validset = [X_train[s] for s in train_idx], [X_train[s] for s in val_idx]
        trainlabel, validlabel = [y_train[s] for s in train_idx], [y_train[s] for s in val_idx]
        train_ls, valid_ls = train(net, trainset, trainlabel, validset, validlabel, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        # train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        # if i == 0:
        #     plot(list(range(1, num_epochs + 1)), [train_ls, valid_ls],
        #          xlabel='epoch', ylabel='rmse', xlim=[1, num_epochs],
        #          legend=['train', 'valid'], yscale='log')
        print(f'折{i + 1}， '
              f'验证log rmse{float(valid_ls[-1]):f}')
    pred = net(test_features)
    test_data["SalePrice"] = torch.squeeze(pred).detach().numpy()
    test_data.to_csv("./data/test.csv")
    return train_l_sum / k, valid_l_sum / k


k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.005, 0, 32
train_l, valid_l = fit(k, train_features, train_labels, num_epochs, lr, weight_decay, batch_size)
# print(f'{k}-折验证: 平均训练log rmse: {float(train_l):f}, '
#       f'平均验证log rmse: {float(valid_l):f}')
