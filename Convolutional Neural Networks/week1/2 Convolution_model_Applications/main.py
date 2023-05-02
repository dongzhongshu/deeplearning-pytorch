import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from matplotlib.pyplot import imread
import scipy
from PIL import Image
import pandas as pd
from cnn_utils import *
# from test_utils import summary, comparator
import torch
from h5dataset import H5Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F

np.random.seed(1)

X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_happy_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# X_train = X_train_orig
# X_test = X_test_orig

# Reshape
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


# index = 124
# plt.imshow(X_train_orig[index]) #display sample training image
# plt.show()


class HappyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=(7, 7), stride=(1, 1), padding=3)
        self.model = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2),
            torch.nn.Flatten(),
            torch.nn.Linear(32768, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        x = self.model(x)
        return x


def happyModel():
    """
    Implements the forward propagation for the binary classification model:
    ZEROPAD2D -> CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code all the values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    None

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)
    """
    model = HappyModel()
    return model


# Print a summary for each layer
# for layer in happy_model.modules():
#     print(layer)

def loss_batch(model, loss_func, xb, yb, opt=None):
    z = model(xb)
    loss = loss_func(z, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit():
    happy_model = happyModel()
    epochs = 100
    batch_size = 16
    learning_rate = 0.001
    dataset = H5Dataset(X_train, Y_train)
    validset = H5Dataset(X_test, Y_test)
    print_cost = True
    testLoader = DataLoader(dataset=dataset, batch_size=batch_size)
    validLoader = DataLoader(dataset=validset, batch_size=batch_size)
    optimizer = torch.optim.SGD(happy_model.parameters(), learning_rate)
    loss_function = torch.nn.BCELoss(reduction='mean')
    validloss = []
    for i in range(epochs):
        happy_model.train()
        for datax, datay in testLoader:
            optimizer.zero_grad()
            inputx = torch.reshape(datax, (-1, 3, datax.shape[1], datax.shape[2])).float()
            forward = happy_model.forward(inputx)
            loss = loss_function(forward, datay.float())
            loss.backward()
            optimizer.step()
        happy_model.eval()
        with torch.no_grad():
            data = [loss_batch(happy_model, loss_function, torch.reshape(xb, (-1, 3, xb.shape[1], xb.shape[2])).float(),
                               yb.float()) for xb, yb in validLoader]
            losses, nums = zip(*data)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        if print_cost == True and i % 10 == 0:
            print("val_loss after epoch %i: %f" % (i, val_loss))
        if print_cost == True and i % 5 == 0:
            validloss.append(val_loss)


# fit()


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_signs_dataset()
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.
Y_train = Y_train_orig.T
Y_test = Y_test_orig.T
# Y_train = convert_to_one_hot(Y_train_orig, 6).T
# Y_test = convert_to_one_hot(Y_test_orig, 6).T
print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))


def getpadding(shape, strides, filter):
    px = math.floor(((shape[0] + 1) * (strides - 1) + filter[0] + 1) / 2)
    py = math.floor(((shape[1] + 1) * (strides - 1) + filter[1] + 1) / 2)
    return px, py


class SignsModel(torch.nn.Module):
    """
    Implements the forward propagation for the model:
    CONV2D -> RELU -> MAXPOOL -> CONV2D -> RELU -> MAXPOOL -> FLATTEN -> DENSE

    Note that for simplicity and grading purposes, you'll hard-code some values
    such as the stride and kernel (filter) sizes.
    Normally, functions should take these values as function parameters.

    Arguments:
    input_img -- input dataset, of shape (input_shape)

    Returns:
    model -- TF Keras model (object containing the information for the entire training process)




    input_img = tf.keras.Input(shape=input_shape)
    ## CONV2D: 8 filters 4x4, stride of 1, padding 'SAME'
    Z1 = tfl.Conv2D(filters= 8. , kernel_size=4 , padding='same',strides=1)(input_img)
    ## RELU
    A1 = tfl.ReLU()(Z1)
    ## MAXPOOL: window 8x8, stride 8, padding 'SAME'
    P1 = tfl.MaxPool2D(pool_size=8, strides=8, padding='SAME')(A1)
    ## CONV2D: 16 filters 2x2, stride 1, padding 'SAME'
    Z2 = tfl.Conv2D(filters= 16. , kernel_size=2 , padding='same',strides=1)(P1)
    ## RELU
    A2 =  tfl.ReLU()(Z2)
    ## MAXPOOL: window 4x4, stride 4, padding 'SAME'
    P2 = tfl.MaxPool2D(pool_size=4, strides=4, padding='SAME')(A2)
    ## FLATTEN
    F = tfl.Flatten()(P2)
    ## Dense layer
    ## 6 neurons in output layer. Hint: one of the arguments should be "activation='softmax'"
    outputs = tfl.Dense(units= 6 , activation='softmax')(F)
    model = tf.keras.Model(inputs=input_img, outputs=outputs)
    """

    def __init__(self, inputshape, paddingmode="same"):
        super().__init__()
        padding = (0, 0)
        kernel_size = (4, 4)
        strides = 1
        if paddingmode == "same":
            padding = getpadding(inputshape, strides, kernel_size)
        #注释代表了思考和测试过程，还是不要删掉吧
        #优化三部曲
        #1、尽可能缩小flatten层数量。
        #2、加入batch normalization
        #3、优化算法改为adam，学习率由0.01改为0.001
        # self.model = torch.nn.Sequential(
        #     torch.nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3),
        #     torch.nn.BatchNorm2d(32),
        #     torch.nn.ReLU(),
        #     torch.nn.MaxPool2d(2),
        #     torch.nn.Flatten(),
        #     torch.nn.Linear(32768, 6),
        # )
        self.conv1 = torch.nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3)
        self.pool = torch.nn.MaxPool2d(8)
        # self.conv1 = torch.nn.Conv2d(3, 8, kernel_size=(3, 3), padding=1,stride=(1,1))
        # self.pool = torch.nn.MaxPool2d(2)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=(7, 7), padding=1, stride=(1,1))
        self.pool2 = torch.nn.MaxPool2d(kernel_size=(4, 4), stride=1)
        # self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=(5, 5), padding=1, stride=(1,1))
        # self.pool3 = torch.nn.MaxPool2d(kernel_size=(4, 4), stride=1)

        self.fc = torch.nn.Linear(64, 6)#！！！最有一层要尽可能的小，否则不会收敛。


        # self.conv1 = torch.nn.Conv2d(3, 10, kernel_size=5)
        # self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        # self.pooling = torch.nn.MaxPool2d(2)
        # self.fc = torch.nn.Linear(3380, 6)





    def forward(self, X):
        # x = self.model(X)
        # return x
        batch_size = X.shape[0]
        # x = F.relu(self.pooling(self.conv1(X)))
        # x = F.relu(self.pooling(self.conv2(x)))
        # x = x.view(batch_size, -1)
        # return self.fc(x)


        x = self.conv1(X)
        # # x = torch.nn.BatchNorm2d(32)(x)
        x = F.relu(torch.nn.BatchNorm2d(32)(x))#加入BATCH NORMALIZATION确实很有效
        # # # x = torch.nn.ReLU()(torch.nn.BatchNorm2d(32)(self.conv1(X)))
        x = self.pool(x)
        x = torch.nn.ReLU()(torch.nn.BatchNorm2d(64)(self.conv2(x)))
        x = self.pool2(x)
        # x = torch.nn.ReLU()((self.conv3(x)))
        # x = self.pool3(x)
        x = x.view(batch_size, -1)
        return self.fc(x)
        # x = self.pool2(torch.nn.ReLU()(torch.nn.BatchNorm2d(16)(self.conv2(x))))
        # x = self.pool3(torch.nn.ReLU()(torch.nn.BatchNorm2d(32)(self.conv3(x))))
        # x = torch.nn.Flatten()(x)
        # x = torch.nn.Linear(x.shape[1], 6).forward(x)
        return x


def fit1(model, epochs=100,
         batch_size=16,
         learning_rate=0.01):
    dataset = H5Dataset(X_train, Y_train)
    validset = H5Dataset(X_test, Y_test)
    print_cost = True
    testLoader = DataLoader(dataset=dataset, batch_size=batch_size)
    validLoader = DataLoader(dataset=validset, batch_size=batch_size)
    loss_func = torch.nn.CrossEntropyLoss()#具体原因见:https://zhuanlan.zhihu.com/p/401117071
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.01)#torch.optim.Adam(model.parameters(), lr=learning_rate)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)#学习率由0.1改为0.001
    validloss = []
    for i in range(epochs):
        model.train()
        for datax, datay in testLoader:
            optimizer.zero_grad()
            inputx = torch.reshape(datax, (-1, 3, datax.shape[1], datax.shape[2])).float()
            x = model(inputx)
            loss = loss_func(x, np.squeeze(datay))#这里要降维，具体原因见:https://blog.csdn.net/weixin_35757704/article/details/119222008
            loss.backward()
            optimizer.step()
            # if print_cost == True and i % 10 == 0:
            #     print("loss after epoch %i: %f" % (i, loss))
        model.eval()
        with torch.no_grad():
            data = [loss_batch(model, loss_func, torch.reshape(xb, (-1, 3, xb.shape[1], xb.shape[2])).float(),
                               np.squeeze(yb)) for xb, yb in validLoader]
            losses, nums = zip(*data)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        if print_cost == True and i % 10 == 0:
            print("val_loss after epoch %i: %f" % (i, val_loss))
        if print_cost == True and i % 5 == 0:
            validloss.append(val_loss)



def test(model, test_loader):
    correct = 0
    total = 0
    correctlist = []
    with torch.no_grad():
        for data in test_loader:
            inputs, lables = data
            outputs = model.forward(inputs)
            _, predict =  torch.max(outputs.data, dim = 1)
            total += lables.size(0)
            correct += (predict == lables).sum().item()
        print('Accuracy on test set: %d %%' % (100 * correct / total))
        correctlist.append(100 * correct / total)


model = SignsModel((64, 64))
fit1(model)
