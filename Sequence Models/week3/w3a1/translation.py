# from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
# from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
# from tensorflow.keras.optimizers import Adam
# from tensorflow.keras.utils import to_categorical
# from tensorflow.keras.models import load_model, Model
# import tensorflow.keras.backend as K
# import tensorflow as tf
import numpy as np

import torch
from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader

m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
Tx = 30
Ty = 10
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)

print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)


# repeator = torch.#RepeatVector(Tx)
# concatenator = Concatenate(axis=-1)
# densor1 = torch.nn.Linear(10, activation = "tanh")
# densor2 = Dense(1, activation = "relu")
# activator = Activation(softmax, name='attention_weights') # We are using a custom softmax(axis = 1) loaded in this notebook
# dotor = Dot(axes = 1)
# def one_step_attention(a, s_prev):
#     """
#     Performs one step of attention: Outputs a context vector computed as a dot product of the attention weights
#     "alphas" and the hidden states "a" of the Bi-LSTM.
#
#     Arguments:
#     a -- hidden state output of the Bi-LSTM, numpy-array of shape (m, Tx, 2*n_a)
#     s_prev -- previous hidden state of the (post-attention) LSTM, numpy-array of shape (m, n_s)
#
#     Returns:
#     context -- context vector, input of the next (post-attention) LSTM cell
#     """
#
#     ### START CODE HERE ###
#     # Use repeator to repeat s_prev to be of shape (m, Tx, n_s) so that you can concatenate it with all hidden states "a" (≈ 1 line)
#     s_prev = s_prev.repeat(s_prev)#repeator(s_prev)
#     # Use concatenator to concatenate a and s_prev on the last axis (≈ 1 line)
#     # For grading purposes, please list 'a' first and 's_prev' second, in this order.
#     concat = torch.cat(a, s_prev)#concatenator([a, s_prev])
#     # Use densor1 to propagate concat through a small fully-connected neural network to compute the "intermediate energies" variable e. (≈1 lines)
#     e = densor1(concat)
#     # Use densor2 to propagate e through a small fully-connected neural network to compute the "energies" variable energies. (≈1 lines)
#     energies = densor2(e)
#     # Use "activator" on "energies" to compute the attention weights "alphas" (≈ 1 line)
#     alphas = activator(energies)
#     # Use dotor together with "alphas" and "a", in this order, to compute the context vector to be given to the next (post-attention) LSTM-cell (≈ 1 line)
#     context = dotor([alphas, a])
#     ### END CODE HERE ###
#
#     return context


class one_step_attention(torch.nn.Module):
    def __init__(self):
        super(one_step_attention, self).__init__()
        self.linear1 = torch.nn.LazyLinear(10)
        self.relu = torch.nn.ReLU()
        self.linear2 = torch.nn.LazyLinear(1)
        self.tanh = torch.nn.Tanh()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, X):
        a, s_prev = X
        s_prev = torch.unsqueeze(s_prev, dim=1).repeat(1, a.shape[1], 1)
        concat = torch.cat([a, s_prev], dim=-1)
        x = self.linear1(concat)
        x = self.tanh(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.softmax(x)
        x = torch.matmul(x.transpose(1, 2),
                         a)  # matmul多维矩阵相乘时，第一维作为批量#https://blog.csdn.net/qsmx666/article/details/105783610
        return x


def one_step_attention_test(target):
    m = 10
    Tx = 30
    n_a = 32
    n_s = 64
    # np.random.seed(10)
    a = np.random.uniform(1, 0, (m, Tx, 2 * n_a)).astype(np.float32)
    s_prev = np.random.uniform(1, 0, (m, n_s)).astype(np.float32) * 1
    context = target((torch.tensor(a), torch.tensor(s_prev)))

    # assert type(context) == tf.python.framework.ops.EagerTensor, "Unexpected type. It should be a Tensor"
    assert tuple(context.shape) == (m, 1, n_s), "Unexpected output shape"
    assert np.all(context.detach().numpy() > 0), "All output values must be > 0 in this example"
    assert np.all(context.detach().numpy() < 1), "All output values must be < 1 in this example"

    # assert np.allclose(context[0][0][0:5].numpy(), [0.50877404, 0.57160693, 0.45448175, 0.50074816, 0.53651875]), "Unexpected values in the result"
    print("\033[92mAll tests passed!")


# one_step_attention_test(one_step_attention())

class modelf(torch.nn.Module):
    def __init__(self, Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size, inputfeatures = 37):
        super(modelf, self).__init__()
        self.onestepattention = one_step_attention()
        self.Tx = Tx
        self.Ty = Ty
        self.n_a = n_a
        self.n_s = n_s
        self.human_vocab_size = human_vocab_size
        self.machine_vocab_size = machine_vocab_size
        self.blstem = torch.nn.LSTM(inputfeatures, self.n_a, bidirectional=True, batch_first=True)
        self.inputfeatures = inputfeatures
        # h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        self.lstmcell = torch.nn.LSTMCell(n_s, n_s)
        self.fc = torch.nn.LazyLinear(len(machine_vocab))
        # self.softmax = torch.nn.Softmax(dim=-1)
        # outputs = []
        outputs = None

    def forward(self, X):
        m = X.shape[0]
        # h_0 = torch.randn(m, 2 , self.hidden_size)
        # c_0 = torch.randn(m, 2 , self.hidden_size)
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        a, b = self.blstem(X)
        # h_n, c_n = b
        outputs = []
        h_n = torch.zeros(m, 2, self.n_a)
        s = torch.zeros(m, self.n_s)
        c = torch.zeros(m, self.n_s)
        for t in range(Ty):
            # Step 2.A: Perform one step of the attention mechanism to get back the context vector at step t (≈ 1 line)
            # h_n = torch.transpose(h_n, 0, 1)
            h_n = h_n.flatten(1)
            context = self.onestepattention((a, h_n))
            context = context.flatten(1)
            # Step 2.B: Apply the post-attention LSTM cell to the "context" vector.
            # Don't forget to pass: initial_state = [hidden state, cell state] (≈ 1 line)
            s, c = self.lstmcell(context, (s, c))
            h_n = s
            # Step 2.C: Apply Dense layer to the hidden state output of the post-attention LSTM (≈ 1 line)
            out = self.fc(s)
            # out = self.softmax(out)
            # outputs = torch.cat(out, dim=0)
            # Step 2.D: Append "out" to the "outputs" list (≈ 1 line)
            outputs.append(out)
        b = torch.stack(outputs, dim=1)
        # b = torch.cat(outputs, dim=0)
        return b


class DateData(Dataset):
    def __init__(self, datax, datay):
        self.datax = datax
        self.datay = datay

    def __getitem__(self, index):
        return self.datax[index], self.datay[index]

    def __len__(self):
        return len(self.datax)

#TODO: 损失下降不够

def fit():
    model = modelf(Tx, Ty, 32, 64, len(human_vocab), len(machine_vocab))
    trainLoader = DataLoader(dataset=DateData(torch.tensor(Xoh, dtype=torch.float32), torch.tensor(Yoh, dtype=torch.float32)), batch_size=32)#数据必须转换为Tensor，否则损失率不会下降。
    opt = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=0.01)
    loss_func = torch.nn.CrossEntropyLoss()
    for i in range(50):
        model.train()
        trainloss = []
        for x, y in trainLoader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y.float())
            trainloss.append(loss.item())
            loss.backward()
            opt.step()

        print(f'for {i}/50: train loss:{np.mean(trainloss)}')

    EXAMPLES = ['3 May 1979', '5 April 09', '21th of August 2016', 'Tue 10 Jul 2007', 'Saturday May 9 2018',
                'March 3 2001', 'March 3rd 2001', '1 March 2001']
    # s00 = np.zeros((1, n_s))
    # c00 = np.zeros((1, n_s))
    for example in EXAMPLES:
        source = string_to_int(example, Tx, human_vocab)
        # print(source)
        source = np.array(list(map(lambda x: to_categorical(x, num_classes=len(human_vocab)), source))).swapaxes(0, 1)
        source = np.swapaxes(source, 0, 1)
        source = np.expand_dims(source, axis=0)
        prediction = model(torch.tensor(source))
        prediction = np.argmax(prediction, axis=-1)
        output = [inv_machine_vocab[int(i)] for i in prediction]
        print("source:", example)
        print("output:", ''.join(output), "\n")

fit()
