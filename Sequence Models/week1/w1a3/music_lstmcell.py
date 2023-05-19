import torch
from torch.nn import LSTMCell
from music21 import *
from grammar import *
from qa import *
from preprocess import *
from music_utils import *
from data_utils import *
from outputs import *
from test_utils import *


class djmodel(torch.nn.Module):
    """
        Implement the djmodel composed of Tx LSTM cells where each cell is responsible
        for learning the following note based on the previous note and context.
        Each cell has the following schema:
                [X_{t}, a_{t-1}, c0_{t-1}] -> RESHAPE() -> LSTM() -> DENSE()
        Arguments:
            Tx -- length of the sequences in the corpus
            LSTM_cell -- LSTM layer instance
            densor -- Dense layer instance
            reshaper -- Reshape layer instance

        Returns:
            model -- a keras instance model with inputs [X, a0, c0]
    """

    def __init__(self, input_size=90, hidden_size=64, Tx=30, batch_size=60):
        super(djmodel, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.cell = LSTMCell(self.input_size, self.hidden_size)
        self.hx = torch.randn((self.batch_size, self.hidden_size))
        self.cx = torch.randn((self.batch_size, self.hidden_size))
        self.fc = torch.nn.Linear(hidden_size, input_size)
        self.Tx = Tx

    def forward(self, X):
        out = []
        for i in range(self.Tx):
            x = X[:, i, :]
            # Step 2.B: Use reshaper to reshape x to be (1, n_values) (≈1 line)
            # x = torch.unsqueeze(x, dim=0)
            # Step 2.C: Perform one step of the LSTM_cell
            hx, cx = self.cell(x, (self.hx, self.cx))
            result = self.fc(hx)
            out.append(result)
        return torch.stack(out)


X, Y, n_values, indices_values, chords = load_music_utils('./data/original_metheny.mid')


def fit():
    model = djmodel()
    opt = torch.optim.Adam(model.parameters())
    loss_func = torch.nn.CrossEntropyLoss()
    model.train()
    dataset = torch.from_numpy(X.astype(np.float32))
    labels = torch.from_numpy(Y.astype(np.float32))
    for i in range(50):
        opt.zero_grad()
        pred = model(dataset)
        loss = loss_func(pred, labels)
        loss.backward()
        opt.step()
        print(f"loss at epoch {i}: {loss}")


# fit()
class music_inference_model(torch.nn.Module):
    def __init__(self, input_size=90, hidden_size=64, Tx=100, batch_size=1):
        super(music_inference_model, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.cell = LSTMCell(self.input_size, self.hidden_size)
        self.hx = torch.randn((self.batch_size, self.hidden_size))
        self.cx = torch.randn((self.batch_size, self.hidden_size))
        self.fc = torch.nn.Linear(hidden_size, input_size)
        self.Tx = Tx
    def forward(self, X):
        outs = []
        for i in range(self.Tx):
            x = X[:, i, :]
            hx, cx = self.cell(x, (self.hx, self.cx))
            result = self.fc(hx)
            result = torch.argmax(result, dim=-1)
            result = torch.nn.Functional.one_hot(result, self.input_size)
            result = torch.unsqueeze(result, dim=1)
            outs.append(result)
        return torch.stack(outs)


n_a = 64
x_initializer = np.zeros((1, 1, n_values))
a_initializer = np.zeros((1, n_a))
c_initializer = np.zeros((1, n_a))


class predict_and_sample(torch.nn.Module):
    """
    Predicts the next value of values using the inference model.

    Arguments:
    inference_model -- Keras model instance for inference time
    x_initializer -- numpy array of shape (1, 1, 90), one-hot vector initializing the values generation
    a_initializer -- numpy array of shape (1, n_a), initializing the hidden state of the LSTM_cell
    c_initializer -- numpy array of shape (1, n_a), initializing the cell state of the LSTM_cel

    Returns:
    results -- numpy-array of shape (Ty, 90), matrix of one-hot vectors representing the values generated
    indices -- numpy-array of shape (Ty, 1), matrix of indices representing the values generated
    """

    def __init__(self, input_size = 90):
        super(predict_and_sample, self).__init__()
        self.input_size = input_size
        self.model = music_inference_model()
    def forward(self, X):
        pred = self.model(X)
        # Step 2: Convert "pred" into an np.array() of indices with the maximum probabilities
        indices = torch.argmax(pred, dim=2)
        # Step 3: Convert indices to one-hot vectors, the shape of the results should be (Ty, n_values)
        results = not torch.nn.Functional.one_hot(indices, self.input_size)
        ### END CODE HERE ###
        return results, indices