# import tensorflow as tf
import numpy as np
import scipy.misc
# from tensorflow.keras.applications.resnet_v2 import ResNet50V2
# from tensorflow.keras.preprocessing import image
# from tensorflow.keras.applications.resnet_v2 import preprocess_input, decode_predictions
# from tensorflow.keras import layers
# from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
# from tensorflow.keras.models import Model, load_model
from resnets_utils import *
# from tensorflow.keras.initializers import random_uniform, glorot_uniform, constant, identity
# from tensorflow.python.framework.ops import EagerTensor
from matplotlib.pyplot import imshow

from torch.nn import Conv2d, MaxPool2d, ReLU, BatchNorm2d
from torchsummary import summary
# from test_utils import summary, comparator
import public_tests
from test_utils import comparator


class identity_block(torch.nn.Module):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to random uniform initializer

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    def __init__(self, inplans, f, filters, training=True, initializer=torch.nn.init.uniform_):
        super(identity_block, self).__init__()
        # Retrieve Filters
        # in pytorch, the input shape is (N, C_{in}, H_{in}, W_{in})
        # X = torch.reshape(X, (X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
        F1, F2, F3 = filters
        self.conv1 = Conv2d(inplans, F1, kernel_size=(1,1), padding="valid")
        self.conv2 = Conv2d(F1, F2, kernel_size=(f,f), padding="same")
        self.conv3 = Conv2d(F2, F3, kernel_size=(1,1), padding="valid")
        self.batchnormal1 = BatchNorm2d(F1)
        self.batchnormal2 = BatchNorm2d(F2)
        self.batchnormal3 = BatchNorm2d(F3)
        self.relu = ReLU()

    def forward(self, X):
        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X
        cache = []
        # First component of main path
        X = self.conv1(X)
        X = self.batchnormal1(X)
        X = self.relu(X)

        ### START CODE HERE
        ## Second component of main path (≈3 lines)
        X = self.conv2(X)
        X = self.batchnormal2(X)
        X = self.relu(X)

        ## Third component of main path (≈2 lines)
        X = self.conv3(X)
        X = self.batchnormal3(X)

        X = torch.add(X_shortcut, X)
        X = self.relu(X)

        return X


# np.random.seed(1)
# X1 = torch.from_numpy(np.ones((1, 4, 4, 3)) * -1)
# X2 = torch.from_numpy(np.ones((1, 4, 4, 3)) * 1)
# X3 = torch.from_numpy(np.ones((1, 4, 4, 3)) * 3)
#
# X = torch.from_numpy(np.concatenate((X1.numpy(), X2.numpy(), X3.numpy()), axis = 0).astype(np.float32))
#
# A3 = identity_block(X, f=2, filters=[4, 4, 3],
#                    initializer=lambda seed=0:torch.tensor(1),
#                    training=False)
# print('\033[1mWith training=False\033[0m\n')
# A3np = A3.detach().numpy()
# print(np.around(A3.detach().numpy()[:,(0,-1),:,:].mean(axis = 3), 5))
# resume = A3np[:,(0,-1),:,:].mean(axis = 3)
# print(resume[1, 1, 0])
#
# print('\n\033[1mWith training=True\033[0m\n')
# np.random.seed(1)
# A4 = identity_block(X, f=2, filters=[3, 3, 3],
#                    initializer=lambda seed=0:torch.tensor(1),
#                    training=True)
# print(np.around(A4.detach().numpy()[:,(0,-1),:,:].mean(axis = 3), 5))
#
# public_tests.identity_block_test(identity_block)


class convolutional_block(torch.nn.Module):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    s -- Integer, specifying the stride to be used
    training -- True: Behave in training mode
                False: Behave in inference mode
    initializer -- to set up the initial weights of a layer. Equals to Glorot uniform initializer,
                   also called Xavier uniform initializer.

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    def __init__(self, inplains, outplains, f, filters, s=2, training=True, initializer=torch.nn.init.normal_):
        super(convolutional_block, self).__init__()
        self.f = f
        self.filters = filters
        self.s = s
        self.training = training
        self.initializer = initializer
        self.F1, self.F2, self.F3 = self.filters
        self.conv1 = Conv2d(inplains, self.F1, stride=(self.s, self.s), kernel_size=(1,1), padding="valid")
        self.conv2 = Conv2d(self.F1, self.F2, kernel_size=(self.f, self.f), padding="same")
        self.conv3 = Conv2d(self.F2, self.F3, kernel_size=(1,1), padding="valid")
        self.convshortcut = Conv2d(inplains, self.F3, kernel_size=(1,1), stride=(self.s, self.s), padding='valid',
                                   )
        self.batchnorm1 = BatchNorm2d(self.F1)
        self.batchnorm2 = BatchNorm2d(self.F2)
        self.batchnorm3 = BatchNorm2d(self.F3)
        self.relu = ReLU()

    # Retrieve Filters
    # in pytorch, the input shape is (N, C_{in}, H_{in}, W_{in})
    # X = torch.reshape(X, (X.shape[0], X.shape[3], X.shape[1], X.shape[2]))
    def forward(self, X):
        # Save the input value. You'll need this later to add back to the main path.
        X_shortcut = X
        cache = []
        # First component of main path
        X = self.conv1(X)
        X = self.batchnorm1(X)
        X = self.relu(X)

        ### START CODE HERE
        ## Second component of main path (≈3 lines)
        X = self.conv2(X)
        X = self.batchnorm2(X)
        X = self.relu(X)

        ## Third component of main path (≈2 lines)
        X = self.conv3(X)
        X = self.batchnorm3(X)

        ##### SHORTCUT PATH ##### (≈2 lines)
        X_shortcut = self.convshortcut(X_shortcut)
        X_shortcut = self.batchnorm3(X_shortcut)

        X = torch.add(X_shortcut, X)
        X = self.relu(X)

        return X


class ResNet50(torch.nn.Module):
    """
    Stage-wise implementation of the architecture of the popular ResNet50:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> FLATTEN -> DENSE

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    def __init__(self):
        super(ResNet50, self).__init__()
        self.input_shape = (1, 3, 64, 64)
        self.classes = 6
        self.zeropad = torch.nn.ZeroPad2d(3)
        self.conv1 = Conv2d(3, 64, (7, 7), stride=(2, 2))
        self.batchnormal = BatchNorm2d(64)
        self.relu = ReLU()
        self.maxpool = MaxPool2d(3, stride=(2, 2))
        self.avgpool = torch.nn.AvgPool2d(2)
        self.module = torch.nn.Sequential(
            convolutional_block(64, 64, 3, filters=[64, 64, 256], s=1),
            identity_block(256, 3, [64, 64, 256]),
            identity_block(256, 3, [64, 64, 256]),
            convolutional_block(256, 128, f=3, filters=[128, 128, 512], s=2),
            identity_block(512, 3, [128, 128, 512]),
            identity_block(512, 3, [128, 128, 512]),
            identity_block(512, 3, [128, 128, 512]),
            convolutional_block(512, 256, f=3, filters=[256, 256, 1024], s=2),
            identity_block(1024, 3, [256, 256, 1024]),
            identity_block(1024, 3, [256, 256, 1024]),
            identity_block(1024, 3, [256, 256, 1024]),
            identity_block(1024, 3, [256, 256, 1024]),
            identity_block(1024, 3, [256, 256, 1024]),
            convolutional_block(1024, 512, f=3, filters=[512, 512, 2048], s=2),
            identity_block(2048, 3, [512, 512, 2048]),
            identity_block(2048, 3, [512, 512, 2048]),
        )
        self.fc = torch.nn.Linear(2048, self.classes)

    def forward(self, X):
        # Define the input as a tensor with shape input_shape
        X_input = X  # torch.randn(self.input_shape)
        # Zero-Padding
        X = self.zeropad(X_input)

        # Stage 1
        X = self.conv1(X)
        X = self.batchnormal(X)
        X = self.relu(X)
        X = self.maxpool(X)

        # Stage 2
        # X = convolutional_block(X, f=3, )
        # X = identity_block(X, 3, )
        # X = identity_block(X, 3, [64, 64, 256])
        #
        # # Stage 3
        # X = convolutional_block(X, f=3, filters=[128, 128, 512], s=2)
        # X = identity_block(X, 3, [128, 128, 512])
        # X = identity_block(X, 3, [128, 128, 512])
        # X = identity_block(X, 3, [128, 128, 512])
        #
        # # Stage 4
        # X = convolutional_block(X, f=3, filters=[256, 256, 1024], s=2)
        # X = identity_block(X, 3, [256, 256, 1024])
        # X = identity_block(X, 3, [256, 256, 1024])
        # X = identity_block(X, 3, [256, 256, 1024])
        # X = identity_block(X, 3, [256, 256, 1024])
        # X = identity_block(X, 3, [256, 256, 1024])
        #
        # # Stage 5
        # X = convolutional_block(X, f=3, filters=[512, 512, 2048], s=2)
        # X = identity_block(X, 3, [512, 512, 2048])
        # X = identity_block(X, 3, [512, 512, 2048])
        X = self.module(X)

        X = self.avgpool(X)
        X = torch.nn.Flatten()(X)
        X = self.fc(X)

        # model = torch.nn.Module(inputs=X_input, outputs=X)
        return X


from torchvision import models

# models.resnet18()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model = models.resnet18()#ResNet50().to(device)
# summary(model, (3, 64, 64), batch_size=-1,  device='cpu')


X_train_orig, Y_train_orig, X_test_orig, Y_test_orig, classes = load_dataset()

# Normalize image vectors
X_train = X_train_orig / 255.
X_test = X_test_orig / 255.

# Convert training and test labels to one hot matrices
Y_train = Y_train_orig.T  # convert_to_one_hot(Y_train_orig, 6).T
Y_test = Y_test_orig.T  # convert_to_one_hot(Y_test_orig, 6).T

print("number of training examples = " + str(X_train.shape[0]))
print("number of test examples = " + str(X_test.shape[0]))
print("X_train shape: " + str(X_train.shape))
print("Y_train shape: " + str(Y_train.shape))
print("X_test shape: " + str(X_test.shape))
print("Y_test shape: " + str(Y_test.shape))
from mydataset import MyDataSet
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ResNet50().to(device)
summary(model, (3, 64, 64), batch_size=-1,  device='cpu')
from torch.utils.data import Dataset, DataLoader


def loss_batch(model, loss_func, xb, yb, opt=None):
    z = model(xb)
    loss = loss_func(z, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(model, print_cost=True):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    traindataset = MyDataSet(X_train, Y_train)
    validdataset = MyDataSet(X_test, Y_test)
    trainLoader = DataLoader(dataset=traindataset,  batch_size=32)
    validLoader = DataLoader(dataset=validdataset,  batch_size=32)
    validloss = []
    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(50):
        model.train()
        for datax, datay in trainLoader:
            optimizer.zero_grad()
            inputx = torch.reshape(datax, (-1, 3, datax.shape[1], datax.shape[2])).float()
            forward = model.forward(inputx)
            loss = loss_func(forward, np.squeeze(datay))
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            data = [loss_batch(model, loss_func, torch.reshape(xb, (-1, 3, xb.shape[1], xb.shape[2])).float(),
                               np.squeeze(yb)) for xb, yb in validLoader]
            losses, nums = zip(*data)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        if print_cost == True:
            print("val_loss after epoch %i: %f" % (i, val_loss))
        if print_cost == True and i % 5 == 0:
            validloss.append(val_loss)




fit(model)
