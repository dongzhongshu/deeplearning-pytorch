from torch.utils.data import Dataset
import torchvision
import torch
import numpy as np

class AlpacaDataset(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label

    def __getitem__(self, index):
        return self.data[index], self.label[index]

    def __len__(self):
        return self.data.shape[0]



normalize = torchvision.transforms.Normalize(
    [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_augs = torchvision.transforms.Compose(
    [torchvision.transforms.RandomResizedCrop(224), torchvision.transforms.RandomHorizontalFlip(),
     torchvision.transforms.ToTensor(), normalize])
test_augs  = torchvision.transforms.Compose([
    torchvision.transforms.Resize(256),
    torchvision.transforms.CenterCrop(224),
    torchvision.transforms.ToTensor(),
    normalize])

train_imgs = torchvision.datasets.ImageFolder("./hotdog/train", transform=train_augs)
test_imgs = torchvision.datasets.ImageFolder("./hotdog/test", transform=test_augs)

model = torchvision.models.mobilenet_v2(pretrained=True)#resnet18(pretrained=True)
for m in model.parameters():
    m.requires_grad = False
print(model)
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
torch.nn.init.xavier_normal(model.classifier[1].weight)


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




def fit():
    trainLoader = torch.utils.data.DataLoader(dataset=train_imgs, batch_size=16, shuffle=True)
    testLoader = torch.utils.data.DataLoader(dataset=test_imgs, batch_size=16, shuffle=True)
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    print_cost = True
    for i in range(10):
        model.train()
        for datax, datay in trainLoader:
            opt.zero_grad()
            pred = model(datax)
            loss = loss_func(pred, datay)
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            data = [loss_batch(model, loss_func, xb,
                               yb) for xb, yb in testLoader]
            losses, nums = zip(*data)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        acc = evaluate_accuracy(model, testLoader)
        if print_cost == True:
            print("val_loss after epoch %i: %f" % (i, val_loss))

fit()

