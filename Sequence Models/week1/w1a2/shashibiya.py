from torch.nn.modules import LSTM
from torch.utils.data import Dataset, DataLoader
import torch
import sys
data = open('shakespeare.txt', 'r').read().lower().split("\n")
words = [word.split() for word in data if word.strip()]
wordstotal = []
for w in words:
    wordstotal.extend(w)
wordstotal.append("\n")
chars = list(set(wordstotal))
data_size, vocab_size = len(data), len(chars)
chars = sorted(chars)
char_to_ix = { ch:i for i,ch in enumerate(chars) }#学习一下
ix_to_char = { i:ch for i,ch in enumerate(chars) }

input_features = len(chars)


class Shashibiya(torch.nn.Module):
    def __init__(self, input_size=input_features, hidden_size=100, num_layers=1, output_size=1, batch_size=16):
        super(Shashibiya, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_directions = 2
        self.lstm = LSTM(input_size, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = torch.nn.Linear(self.hidden_size * self.num_directions, input_features)
    def forward(self, input):
        # batch_size, seq_len = input.shape[0], input.shape[1]
        # batch_size=16
        # h_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
        # c_0 = torch.randn(self.num_directions * self.num_layers, batch_size, self.hidden_size)
         output, _ = self.lstm(input)
        res = self.fc(output.data)
        return res


"""
每一句话为一个数据
"""
import torch.nn.utils.rnn as rnn_utils
def collate_fn(train_data):
    train_data.sort(key=lambda x: len(x[0]), reverse=True)
    data = [sq[0] for sq in train_data]
    label = [sq[1] for sq in train_data]
    data_length = [len(sq) for sq in data]
    data = rnn_utils.pad_sequence(data, batch_first=True, padding_value=0.0)  # 用零补充，使长度对齐
    label = rnn_utils.pad_sequence(label, batch_first=True, padding_value=0.0)  # 这行代码只是为了把列表变为tensor
    return data, label, data_length


import torch.nn.functional as F
class LableSet(Dataset):
    def __init__(self, datax, datay):
        self.datax = datax
        self.datay = datay
    def __getitem__(self, index):
        line = self.datax[index]
        x = []
        y = []
        for i, w in enumerate(line):
            xidx = torch.tensor(F.one_hot(torch.tensor(char_to_ix[w]), input_features),  dtype=torch.float32)
            x.append(xidx)
            if i == len(line)-1:
                ylabel = "\n"
            else:
                ylabel = line[i+1]
            yidx = torch.tensor(char_to_ix[ylabel])#F.one_hot(torch.tensor(char_to_ix[ylabel]), input_features)
            y.append(yidx)#F.one_hot(yidx, input_features)
        return torch.stack(x), torch.stack(y)#不能直接用tensor转换，用stack。https://blog.csdn.net/liu16659/article/details/114752918
    def __len__(self):
        return len(self.datax)


import numpy as np
def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    # preds = np.asarray(preds).astype('float64')
    # preds = np.log(preds) / temperature
    # exp_preds = np.exp(preds)
    # preds = exp_preds / np.sum(exp_preds)
    # probas = np.random.multinomial(1, preds, 1)
    out = np.random.choice(range(input_features), p = preds.ravel())
    # out = np.random.choice(range(3))
    return out


def fit():
    model = Shashibiya()
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())
    testset = LableSet(words, None)
    testLoader = DataLoader(dataset=testset, batch_size=16, collate_fn=collate_fn)
    for i in range(50):
        model.train()
        loss = None
        for datax, datay, batch_x_len in testLoader:
            opt.zero_grad()
            batch_x_pack = rnn_utils.pack_padded_sequence(datax, batch_x_len, batch_first=True)
            pred = model(batch_x_pack)
            batch_y_pack = rnn_utils.pack_padded_sequence(datay, batch_x_len, batch_first=True)
            loss = loss_func(pred, batch_y_pack.data)
            loss.backward()
            opt.step()
        print(f'Epoch {i} train loss: {loss.item()}')

    model.eval()
    torch.save(model, "shashibiya.pth")
    generated = ''
    # sentence = text[start_index: start_index + Tx]
    # sentence = '0'*Tx
    Tx = 40
    usr_input = input("Write the beginning of your poem, the Shakespeare machine will complete it. Your input is: ")
    # zero pad the sentence to Tx characters.
    sentence = ('{0:0>' + str(Tx) + '}').format(usr_input).lower()
    generated += usr_input

    sys.stdout.write("\n\nHere is your poem: \n\n")
    sys.stdout.write(usr_input)
    w = usr_input
    for i in range(400):
        x_pred = torch.tensor(F.one_hot(torch.tensor(char_to_ix[w]), input_features), requires_grad=True, dtype=torch.float32)
        x_pred = torch.unsqueeze(x_pred, 0).unsqueeze(0)
        preds = model(x_pred)
        preds = torch.softmax(preds, dim=2).squeeze()
        # preds, indexes = torch.sort(preds, descending=True)
        # preds = preds[:3]
        next_index = sample(preds.detach().numpy(), temperature=1.0)
        # value = indexes[next_index]value.item()
        next_char = ix_to_char[next_index]

        # generated += next_char + " "
        # sentence = sentence[1:] + next_char

        sys.stdout.write(" " + next_char + " ")
        sys.stdout.flush()
        w = next_char
        if next_char == '\n':
            continue



fit()