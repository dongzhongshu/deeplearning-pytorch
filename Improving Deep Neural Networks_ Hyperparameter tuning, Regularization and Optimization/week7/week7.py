import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import h5py
import numpy as np
import torch
# import tensorflow as tf
import matplotlib.pyplot as plt
# from tensorflow.python.framework.ops import EagerTensor
# from tensorflow.python.ops.resource_variable_ops import
from h5dataset import H5Dataset
from torch.autograd import Variable
import time

train_dataset = h5py.File('datasets/train_signs.h5', "r")
test_dataset = h5py.File('datasets/test_signs.h5', "r")

x_train = H5Dataset(train_dataset['train_set_x'], train_dataset['train_set_y'])
# x_train = torch.from_numpy(train_dataset['train_set_x'])
# y_train = torch.from_numpy(train_dataset['train_set_y'])

x_test = H5Dataset(test_dataset['test_set_x'], test_dataset['test_set_y'])
# x_test = torch.from_numpy(test_dataset['test_set_x'])
# y_test = torch.from_numpy(test_dataset['test_set_y'])

from torch.utils.data import DataLoader

print(len(x_train))


def normalize(image):
    """
    Transform an image into a tensor of shape (64 * 64 * 3, 1)
    and normalize its components.

    Arguments
    image - Tensor.

    Returns:
    result -- Transformed tensor
    """
    image = torch.divide(image, 256.)
    # image = torch.reshape(image, (-1, 1))
    image = torch.reshape(image, (image.shape[0], -1))
    return image


for element, label in x_train:
    d = np.array(element)
    t = torch.from_numpy(d)
    print(normalize(t))
    break


def linear_function():
    """
    Implements a linear function:
            Initializes X to be a random tensor of shape (3,1)
            Initializes W to be a random tensor of shape (4,3)
            Initializes b to be a random tensor of shape (4,1)
    Returns:
    result -- Y = WX + b
    """

    np.random.seed(1)

    """
    Note, to ensure that the "random" numbers generated match the expected results,
    please create the variables in the order given in the starting code below.
    (Do not re-arrange the order).
    """
    # (approx. 4 lines)
    # X = ...
    # W = ...
    # b = ...
    # Y = ...
    # YOUR CODE STARTS HERE
    # X = tf.constant(np.random.randn(3, 1))
    # W = tf.constant(np.random.randn(4, 3))
    # b = tf.constant(np.random.randn(4, 1))
    # Y = tf.add(tf.matmul(W, X), b)
    X = torch.rand((3, 1))
    W = torch.rand((4, 3))
    b = torch.rand((4, 1))
    a = torch.mm(W, X)  # 不是torch.mul
    Y = torch.add(a, b)

    # YOUR CODE ENDS HERE
    return Y


result = linear_function()
print(result)


def sigmoid(z):
    """
    Computes the sigmoid of z

    Arguments:
    z -- input value, scalar or vector

    Returns:
    a -- (tf.float32) the sigmoid of z
    """
    # tf.keras.activations.sigmoid requires float16, float32, float64, complex64, or complex128.

    # (approx. 2 lines)
    # z = ...
    # result = ...
    # YOUR CODE STARTS HERE
    a = torch.sigmoid(z)
    return a


result = sigmoid(torch.tensor(-1))
print("type: " + str(type(result)))
print("dtype: " + str(result.dtype))
print("sigmoid(-1) = " + str(result))

from torch.nn.functional import one_hot


# GRADED FUNCTION: one_hot_matrix
def one_hot_matrix(label, depth=6):
    """
    Computes the one hot encoding for a single label
    
    Arguments:
        label --  (int) Categorical labels
        depth --  (int) Number of different classes that label can take
    
    Returns:
         one_hot -- tf.Tensor A single-column matrix with the one hot encoding.
    """
    # (approx. 1 line)
    # one_hot = ...
    # YOUR CODE STARTS HERE
    t = one_hot(label, depth)
    return t


print(one_hot_matrix(torch.tensor([[1], [2], [2]]), 6))


def initialize_parameters():
    """
    Initializes parameters to build a neural network with TensorFlow. The shapes are:
                        W1 : [25, 12288]
                        b1 : [25, 1]
                        W2 : [12, 25]
                        b2 : [12, 1]
                        W3 : [6, 12]
                        b3 : [6, 1]

    Returns:
    parameters -- a dictionary of tensors containing W1, b1, W2, b2, W3, b3
    """

    # initializer = tf.keras.initializers.GlorotNormal(seed=1)
    # # (approx. 6 lines of code)
    # W1 = tf.Variable(initializer(shape=(25, 12288)))
    # b1 = tf.Variable(initializer(shape=(25, 1)))
    # W2 = tf.Variable(initializer(shape=(12, 25)))
    # b2 = tf.Variable(initializer(shape=(12, 1)))
    # W3 = tf.Variable(initializer(shape=(6, 12)))
    # b3 = tf.Variable(initializer(shape=(6, 1)))
    # YOUR CODE STARTS HERE

    # YOUR CODE ENDS HERE
    W1 = torch.nn.init.normal_(torch.zeros(25, 12288))
    b1 = torch.zeros((25, 1))
    W2 = torch.nn.init.normal(torch.zeros(12, 25))
    b2 = torch.zeros((12, 1))
    W3 = torch.nn.init.normal(torch.zeros(6, 12))
    b3 = torch.zeros((6, 1))
    # parameters = {"W1": W1,
    #               "b1": b1,
    #               "W2": W2,
    #               "b2": b2,
    #               "W3": W3,
    #               "b3": b3}
    # 一定要用Variable，否则参数不进行更新。
    parameters = {"W1": Variable(W1, requires_grad=True),
                  "b1": Variable(b1, requires_grad=True),
                  "W2": Variable(W2, requires_grad=True),
                  "b2": Variable(b2, requires_grad=True),
                  "W3": Variable(W3, requires_grad=True),
                  "b3": Variable(b3, requires_grad=True)}

    return parameters


parameters = initialize_parameters()
print(parameters)


def forward_propagation(X, parameters):
    """
    Implements the forward propagation for the model: LINEAR -> RELU -> LINEAR -> RELU -> LINEAR

    Arguments:
    X -- input dataset placeholder, of shape (input size, number of examples)
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3"
                  the shapes are given in initialize_parameters

    Returns:
    Z3 -- the output of the last LINEAR unit
    """

    # Retrieve the parameters from the dictionary "parameters"
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    Z1 = torch.mm(W1, X.T) + b1
    A1 = torch.nn.functional.relu(Z1)
    Z2 = torch.mm(W2, A1) + b2
    A2 = torch.nn.functional.relu(Z2)
    Z3 = torch.mm(W3, A2) + b3
    return Z3


def compute_cost(logits, labels):
    """
    Computes the cost

    Arguments:
    logits -- output of forward propagation (output of the last LINEAR unit), of shape (6, number of examples)
    labels -- "true" labels vector, same shape as Z3

    Returns:
    cost - Tensor of the cost function
    """
    cost = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
    return cost


def model(dataset, learning_rate=0.0001,
          num_epochs=1500, minibatch_size=32, print_cost=True):
    """
    Implements a three-layer tensorflow neural network: LINEAR->RELU->LINEAR->RELU->LINEAR->SOFTMAX.

    Arguments:
    X_train -- training set, of shape (input size = 12288, number of training examples = 1080)
    Y_train -- test set, of shape (output size = 6, number of training examples = 1080)
    X_test -- training set, of shape (input size = 12288, number of training examples = 120)
    Y_test -- test set, of shape (output size = 6, number of test examples = 120)
    learning_rate -- learning rate of the optimization
    num_epochs -- number of epochs of the optimization loop
    minibatch_size -- size of a minibatch
    print_cost -- True to print the cost every 100 epochs

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    costs = []
    parameters = initialize_parameters()
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']
    W3 = parameters['W3']
    b3 = parameters['b3']
    optimizer = torch.optim.SGD([W1, b1, W2, b2, W3, b3], learning_rate)
    # X_train = X_train.batch(minibatch_size, drop_remainder=True).prefetch(8)  # <<< extra step
    # Y_train = Y_train.batch(minibatch_size, drop_remainder=True).prefetch(8)  # loads memory faster
    trainloader = DataLoader(dataset=dataset, batch_size=minibatch_size)
    for epoch in range(num_epochs):
        epoch_cost = 0.
        for minibatch_X, minibatch_Y in trainloader:
            optimizer.zero_grad()
            minibatch_X = normalize(minibatch_X)
            minibatch_Y = one_hot_matrix(minibatch_Y)
            Z3 = forward_propagation(minibatch_X, parameters)
            minibatch_cost = compute_cost(Z3.T, minibatch_Y.float())
            minibatch_cost.backward()
            optimizer.step()
            epoch_cost += minibatch_cost / minibatch_size

        if print_cost == True and epoch % 10 == 0:
            print("Cost after epoch %i: %f" % (epoch, epoch_cost))
        if print_cost == True and epoch % 5 == 0:
            costs.append(torch.tensor(epoch_cost).numpy())
    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Save the parameters in a variable
    print("Parameters have been trained!")

    return parameters


# model(x_train, num_epochs=200)

"""
简化版本
"""


class BinaryCrossModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Sequential(
            torch.nn.Linear(12288, 25),
            torch.nn.ReLU(True),
            torch.nn.Linear(25, 12),
            torch.nn.ReLU(True),
            torch.nn.Linear(12, 6),
            # torch.nn.ReLU(True),
            # torch.nn.Linear(6, 1)
        )

    def forward(self, X):
        return self.layer(X)


def loss_batch(model, loss_func, xb, yb, opt=None):
    z = model(xb)
    loss = loss_func(z, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)


def fit(dataset, validset, learning_rate=0.0001,
        num_epochs=1500, minibatch_size=32, print_cost=True):
    model = BinaryCrossModel()
    optimizer = torch.optim.SGD(model.parameters(), learning_rate)
    trainloader = DataLoader(dataset=dataset, batch_size=minibatch_size)
    validloader = DataLoader(dataset=validset, batch_size=minibatch_size)
    loss_func = torch.nn.functional.cross_entropy
    validloss = []
    for i in range(num_epochs):
        # 我们总是在训练之前调用model.train()，并在推理之前调用model.eval()，因为诸如nn.BatchNorm2d和nn.Dropout之类的层会使用它们，以确保这些不同阶段的行为正确
        model.train()
        for minibatch_X, minibatch_Y in trainloader:
            minibatch_X = normalize(minibatch_X)
            # minibatch_Y = one_hot_matrix(minibatch_Y)
            loss_batch(model, loss_func, minibatch_X, minibatch_Y, optimizer)
        model.eval()
        with torch.no_grad():
            data = [loss_batch(model, loss_func, normalize(xb), yb) for xb, yb in validloader]
            losses, nums = zip(*data)
        val_loss = np.sum(np.multiply(losses, nums)) / np.sum(nums)
        if print_cost == True and i % 10 == 0:
            print("Cost after epoch %i: %f" % (i, val_loss))
        if print_cost == True and i % 5 == 0:
            validloss.append(val_loss)

    plt.plot(np.squeeze(validloss))
    plt.ylabel('cost')
    plt.xlabel('iterations (per fives)')
    plt.title("Learning rate =" + str(learning_rate))
    plt.show()

    # Save the parameters in a variable
    print("Parameters have been trained!")

    return parameters


fit(x_train, x_test)
