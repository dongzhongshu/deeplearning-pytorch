import numpy as np
import torch
from emo_utils import *
import emoji
import matplotlib.pyplot as plt
from test_utils import *

X_train, Y_train = read_csv('data/train_emoji.csv')
X_test, Y_test = read_csv('data/tesss.csv')

Y_oh_train = convert_to_one_hot(Y_train, C=5)
Y_oh_test = convert_to_one_hot(Y_test, C=5)
word_to_index, index_to_word, word_to_vec_map = read_glove_vecs('./data/glove.6B.50d.txt')


def sentence_to_avg(sentence, word_to_vec_map):
    """
    Converts a sentence (string) into a list of words (strings). Extracts the GloVe representation of each word
    and averages its value into a single vector encoding the meaning of the sentence.

    Arguments:
    sentence -- string, one training example from X
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation

    Returns:
    avg -- average vector encoding information about the sentence, numpy-array of shape (50,)
    """
    # Get a valid word contained in the word_to_vec_map.
    any_word = list(word_to_vec_map.keys())[0]
    ### START CODE HERE ###
    # Step 1: Split sentence into list of lower case words (≈ 1 line)
    words = sentence.lower().split()
    # Initialize the average word vector, should have the same shape as your word vectors.
    avg = np.zeros(word_to_vec_map[any_word].shape)
    # Initialize count to 0
    count = 0

    # Step 2: average the word vectors. You can loop over the words in the list "words".
    for w in words:
        if w in list(word_to_vec_map.keys()):
            avg += word_to_vec_map[w]
            count += 1

    if count > 0:
        avg = avg / count
    return avg


def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=200):
    """
    Model to train word vector representations in numpy.

    Arguments:
    X -- input data, numpy array of sentences as strings, of shape (m, 1)
    Y -- labels, numpy array of integers between 0 and 7, numpy-array of shape (m, 1)
    word_to_vec_map -- dictionary mapping every word in a vocabulary into its 50-dimensional vector representation
    learning_rate -- learning_rate for the stochastic gradient descent algorithm
    num_iterations -- number of iterations

    Returns:
    pred -- vector of predictions, numpy-array of shape (m, 1)
    W -- weight matrix of the softmax layer, of shape (n_y, n_h)
    b -- bias of the softmax layer, of shape (n_y,)
    """

    # Get a valid word contained in the word_to_vec_map
    any_word = list(word_to_vec_map.keys())[0]

    # Initialize cost. It is needed during grading
    cost = 0

    # Define number of training examples
    m = Y.shape[0]  # number of training examples
    n_y = len(np.unique(Y))  # number of classes
    n_h = word_to_vec_map[any_word].shape[0]  # dimensions of the GloVe vectors

    # Initialize parameters using Xavier initialization
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # Convert Y to Y_onehot with n_y classes
    Y_oh = convert_to_one_hot(Y, C=n_y)

    # Optimization loop
    for t in range(num_iterations):  # Loop over the number of iterations
        for i in range(m):  # Loop over the training examples

            ### START CODE HERE ### (≈ 4 lines of code)
            # Average the word vectors of the words from the i'th training example
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # Forward propagate the avg through the softmax layer
            z = np.add(np.dot(W, avg), b)
            a = softmax(z)

            # Compute cost using the i'th training label's one hot representation and "A" (the output of the softmax)
            cost = -np.sum(np.dot(Y_oh[i], np.log(a)))
            ### END CODE HERE ###

            # Compute gradients
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # Update parameters with Stochastic Gradient Descent
            W = W - learning_rate * dW
            b = b - learning_rate * db

        if t % 10 == 0:
            print("Epoch: " + str(t) + " --- cost = " + str(cost))
            pred = predict(X, Y, W, b, word_to_vec_map)  # predict is defined in emo_utils.py

    return pred, W, b


def test():
    np.random.seed(1)
    pred, W, b = model(X_train, Y_train, word_to_vec_map)
    print(pred)
    print("Training set:")
    pred_train = predict(X_train, Y_train, W, b, word_to_vec_map)
    print('Test set:')
    pred_test = predict(X_test, Y_test, W, b, word_to_vec_map)


# def predict_single(sentence, W=W, b=b, word_to_vec_map=word_to_vec_map):
#     """
#     Given X (sentences) and Y (emoji indices), predict emojis and compute the accuracy of your model over the given set.
#
#     Arguments:
#     X -- input data containing sentences, numpy array of shape (m, None)
#     Y -- labels, containing index of the label emoji, numpy array of shape (m, 1)
#
#     Returns:
#     pred -- numpy array of shape (m, 1) with your predictions
#     """
#
#     any_word = list(word_to_vec_map.keys())[0]
#     # number of classes
#     n_h = word_to_vec_map[any_word].shape[0]
#
#     # Split jth test example (sentence) into list of lower case words
#     words = sentence.lower().split()
#
#     # Average words' vectors
#     avg = np.zeros((n_h,))
#     count = 0
#     for w in words:
#         if w in word_to_vec_map:
#             avg += word_to_vec_map[w]
#             count += 1
#
#     if count > 0:
#         avg = avg / count
#
#     # Forward propagation
#     Z = np.dot(W, avg) + b
#     A = softmax(Z)
#     pred = np.argmax(A)
#
#     return pred


#####################################V2-LSTM#######################################
import numpy as np

# import tensorflow
# np.random.seed(0)
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
# from tensorflow.keras.layers import Embedding
# from tensorflow.keras.preprocessing import sequence
# from tensorflow.keras.initializers import glorot_uniform
np.random.seed(1)


def sentences_to_indices(X, word_to_index, max_len):
    """
    Converts an array of sentences (strings) into an array of indices corresponding to words in the sentences.
    The output shape should be such that it can be given to `Embedding()` (described in Figure 4).

    Arguments:
    X -- array of sentences (strings), of shape (m, 1)
    word_to_index -- a dictionary containing the each word mapped to its index
    max_len -- maximum number of words in a sentence. You can assume every sentence in X is no longer than this.

    Returns:
    X_indices -- array of indices corresponding to words in the sentences from X, of shape (m, max_len)
    """
    m = X.shape[0]  # number of training examples

    ### START CODE HERE ###
    # Initialize X_indices as a numpy matrix of zeros and the correct shape (≈ 1 line)
    X_indices = np.zeros([m, max_len])
    for i in range(m):  # loop over training examples

        # Convert the ith training sentence in lower case and split is into words. You should get a list of words.
        sentence_words = X[i].lower().split()

        # Initialize j to 0
        j = 0

        # Loop over the words of sentence_words

        for w in sentence_words:
            # if w exists in the word_to_index dictionary
            if w in word_to_index:
                # Set the (i,j)th entry of X_indices to the index of the correct word.
                X_indices[i, j] = word_to_index[w]
                # Increment j to j + 1
                j = j + 1
        return X_indices


def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    Creates a Keras Embedding() layer and loads in pre-trained GloVe 50-dimensional vectors.

    Arguments:
    word_to_vec_map -- dictionary mapping words to their GloVe vector representation.
    word_to_index -- dictionary mapping from words to their indices in the vocabulary (400,001 words)

    Returns:
    embedding_layer -- pretrained layer Keras instance
    """

    vocab_size = len(word_to_index) + 1  # adding 1 to fit Keras embedding (requirement)
    any_word = list(word_to_vec_map.keys())[0]
    emb_dim = word_to_vec_map[any_word].shape[0]  # define dimensionality of your GloVe word vectors (= 50)

    ### START CODE HERE ###
    # Step 1
    # Initialize the embedding matrix as a numpy array of zeros.
    # See instructions above to choose the correct shape.
    emb_matrix = np.zeros([vocab_size, emb_dim])

    # Step 2
    # Set each row "idx" of the embedding matrix to be
    # the word vector representation of the idx'th word of the vocabulary
    for word, idx in word_to_index.items():
        emb_matrix[idx, :] = word_to_vec_map[word]

    embedding_layer = torch.nn.Embedding(vocab_size, emb_dim)
    embedding_layer.from_pretrained(torch.from_numpy(emb_matrix))
    return embedding_layer


# embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

class Emojify_V2(torch.nn.Module):
    def __init__(self, input_features=50, hidden_size=128, maxlen=10):
        super(Emojify_V2, self).__init__()
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.maxlen = maxlen
        self.lstm = torch.nn.LSTM(input_features, hidden_size, num_layers=2, batch_first=True, dropout=0.5  ,
                                  bidirectional=False)
        self.embedding = pretrained_embedding_layer(word_to_vec_map, word_to_index)
        # self.dropout = torch.nn.Dropout()
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.1),
            torch.nn.Linear(self.hidden_size * 1 , 5))

    def forward(self, X):
        embedding = self.embedding(X)
        x, _ = self.lstm(embedding)
        # x = x.reshape(-1, self.hidden_size * 2)
        # x = x.reshape((x.shape[0], -1))
        # x = self.dropout(x)
        x = x[:, -1, :]
        # x = self.dropout(x)
        x = self.fc(x)
        # x = x.reshape((x.shape[0], -1))
        return x


maxLen = len(max(X_train, key=len).split())
X_train_indices = torch.tensor(sentences_to_indices(X_train, word_to_index, maxLen), dtype=torch.int)
Y_train_oh = torch.tensor(Y_train, dtype=torch.int64)  #convert_to_one_hot(Y_train, C=5)

X_test_indices = torch.tensor(sentences_to_indices(X_test, word_to_index, max_len=maxLen), dtype=torch.int)
Y_test_oh = torch.tensor(Y_train, dtype=torch.int64)#convert_to_one_hot(Y_test, C=5)


class EmojiDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def fit():
    model = Emojify_V2()
    print(model)
    trainset = EmojiDataset(X_train_indices, Y_train_oh)
    trainLoader = torch.utils.data.DataLoader(dataset=trainset, batch_size=32, shuffle=True)
    opt = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_func = torch.nn.CrossEntropyLoss()
    testset = EmojiDataset(X_test_indices, Y_test_oh)
    testLoader = torch.utils.data.DataLoader(dataset=testset, batch_size=32)

    for i in range(50):
        train_loss = []
        train_correct = []
        model.train()
        for x, y in trainLoader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_func(pred, y)
            loss.backward()
            opt.step()
            train_loss.append(loss.item())
            a = torch.argmax(pred, dim = -1)
            b = torch.mean(torch.tensor(a == y, dtype=torch.float32))
            train_correct.append(b.item())
            # train_correct += torch.sum()
        # print(f'Epoch {i} train loss: {loss.item()}')
        model.eval()
        val_loss = []
        val_correct = []
        with torch.no_grad():
            for x, y in testLoader:
                pred = model(x)
                loss = loss_func(pred, y)
                val_loss.append(loss.item())
                a = torch.argmax(pred, dim = -1)
                b = torch.mean(torch.tensor(a == y, dtype=torch.float32))
                val_correct.append(b.item())
                # val_correct += torch.sum(torch.tensor(a == y, dtype=torch.float32))
                # print("accuracy:", torch.mean(a))
        print(f'Epoch {i}/50 --- train loss {np.round(np.mean(train_loss), 5)} --- val loss {np.round(np.mean(val_loss), 5)}, '
              f'train accuracy {np.round(np.mean(train_correct), 5)} --- val accuracy {np.round(np.mean(val_correct), 5)}')

fit()
