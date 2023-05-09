import torch
import pandas as pd
from PIL import Image

from torch.utils.data import Dataset, DataLoader
import torchvision
import numpy as np

datalabels = pd.read_csv("data/train.csv")
labels = sorted(set(datalabels.iloc[:, 1]))
key_to_value = dict([(v, i) for i, v in enumerate(labels)])
value_to_key = dict([(i, v) for i, v in enumerate(labels)])

normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose(
    [torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor(), normalize])
test_augs = torchvision.transforms.Compose(
    [torchvision.transforms.RandomResizedCrop(224),
     torchvision.transforms.ToTensor()])


def Myloader(path):
    return Image.open(path).convert('RGB')


class LeavesDataset(Dataset):
    def __init__(self, transform, loader, mode='train', valid_ratio=0.2):
        self.data = pd.read_csv("data/train.csv")
        self.transform = transform
        self.loader = loader
        # 计算 length
        self.data_len = len(self.data.index) - 1
        # print(self.data_len)
        self.train_len_ = int(self.data_len * (1 - valid_ratio))
        if mode == "train":
            self.len_ = self.train_len_
            self.image_arr = np.asarray(self.data.iloc[:self.len_, 0])
            self.label_arr = np.asarray(self.data.iloc[:self.len_, 1])
        else:
            self.len_ = self.data_len - self.train_len_
            self.image_arr = np.asarray(self.data.iloc[self.train_len_:, 0])
            self.label_arr = np.asarray(self.data.iloc[self.train_len_:, 1])

    def __getitem__(self, index):
        imagepath = "./data/" + self.image_arr[index]
        label = key_to_value[self.label_arr[index]]
        image = self.loader(imagepath)
        image = self.transform(image)
        return image, label

    def __len__(self):
        return self.len_





def loss_batch(model, loss_func, xb, yb, opt=None):
    z = model(xb)
    loss = loss_func(z, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def accuracy(y_hat, y):
    """Compute the number of correct predictions."""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = np.argmax(y_hat, axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(np.sum(cmp.type(y.dtype).numpy()))


def evaluate_accuracy(net, data_iter):
    """Compute the accuracy for a model on a dataset."""
    if isinstance(net, torch.nn.Module):
        net.eval()  # Set the model to evaluation mode
    # metric = Accumulator(2)  # No. of correct predictions, no. of predictions

    with torch.no_grad():
        accnum = 0.0
        totalnum = 0.0
        for X, y in data_iter:
            acc = accuracy(net(X), y)
            accnum += acc
            total = y.numel()
            totalnum += total
    return accnum / totalnum


def fit(model):
    optimizer = torch.optim.Adam(model.parameters())
    batch_size = 16
    loss_func = torch.nn.CrossEntropyLoss()
    losses = []
    epochs = 10
    traindataset = LeavesDataset(train_augs, Myloader)
    trainLoader = DataLoader(dataset=traindataset, batch_size=batch_size, shuffle=True)
    testdataset = LeavesDataset(test_augs, Myloader, "test")
    testLoader = DataLoader(dataset=testdataset, batch_size=batch_size, shuffle=True)
    print_cost = True
    for i in range(epochs):
        model.train()
        for datax, datay in trainLoader:
            optimizer.zero_grad()
            forward = model(datax)
            loss = loss_func(forward, datay)
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            data = [loss_batch(model, loss_func,  x, y) for x, y in testLoader]
            loss_items, nums = zip(*data)
        val_loss = np.sum(np.multiply(loss_items, nums)) / np.sum(nums)
        acc = evaluate_accuracy(model, testLoader)
        if print_cost == True:
            print("val_loss after epoch %i: %f, acc: %f" % (i, val_loss, acc))


modelclassify = torchvision.models.resnet18(pretrained=True)#torchvision.models.mobilenet_v2(pretrained=True)
for m in modelclassify.parameters():
    m.requires_grad = False
# modelclassify.classifier[1] = torch.nn.Linear(modelclassify.classifier[1].in_features, len(labels))
modelclassify.fc = torch.nn.Linear(modelclassify.fc.in_features, len(labels))
torch.nn.init.xavier_normal(modelclassify.fc.weight)
fit(modelclassify)