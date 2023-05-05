from torch.utils.data import Dataset


class MyDataSet(Dataset):
    def __init__(self, x_train, y_train):
        self.x_train = x_train
        self.y_train = y_train

    def __getitem__(self, index):
        return self.x_train[index], self.y_train[index]
    def __len__(self):
        return self.x_train.shape[0]