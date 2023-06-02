# import tensorflow as tf
import pandas as pd
import time
import numpy as np
import torch
import matplotlib.pyplot as plt

# from tensorflow.keras.layers import Embedding, MultiHeadAttention, Dense, Input, Dropout, LayerNormalization
from transformers import DistilBertTokenizerFast  # , TFDistilBertModel
from transformers import TFDistilBertForTokenClassification
from tqdm import tqdm_notebook as tqdm


def get_angles(pos, i, d):
    """
    Get the angles for the positional encoding

    Arguments:
        pos -- Column vector containing the positions [[0], [1], ...,[N-1]]
        i --   Row vector containing the dimension span [[0, 1, 2, ..., M-1]]
        d(integer) -- Encoding size

    Returns:
        angles -- (pos, d) numpy array
    """
    # START CODE HERE
    angles = pos / (np.power(10000, (2 * (i // 2)) / np.float32(d)))
    # END CODE HERE

    return angles


def positional_encoding(positions, d):
    """
    Precomputes a matrix with all the positional encodings

    Arguments:
        positions (int) -- Maximum number of positions to be encoded
        d (int) -- Encoding size

    Returns:
        pos_encoding -- (1, position, d_model) A matrix with the positional encodings
    """
    # START CODE HERE
    # initialize a matrix angle_rads of all the angles
    angle_rads = get_angles(np.arange(positions)[:, np.newaxis],
                            np.arange(d)[np.newaxis, :],
                            d)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    pos_encoding = angle_rads[np.newaxis, ...]
    return torch.tensor(pos_encoding, dtype=torch.float32)




def get_angles_test(target):
    position = 4
    d_model = 16
    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    result = target(pos_m, dims, d_model)

    assert type(result) == np.ndarray, "You must return a numpy ndarray"
    assert result.shape == (position, d_model), f"Wrong shape. We expected: ({position}, {d_model})"
    assert np.sum(result[0, :]) == 0
    assert np.isclose(np.sum(result[:, 0]), position * (position - 1) / 2)
    even_cols =  result[:, 0::2]
    odd_cols = result[:,  1::2]
    assert np.all(even_cols == odd_cols), "Submatrices of odd and even columns must be equal"
    limit = (position - 1) / np.power(10000,14.0/16.0)
    assert np.isclose(result[position - 1, d_model -1], limit ), f"Last value must be {limit}"

    print("\033[92mAll tests passed")


# get_angles_test(get_angles)


def positional_encoding_test(target):
    position = 8
    d_model = 16

    pos_encoding = target(position, d_model)
    sin_part = pos_encoding[:, :, 0::2]
    cos_part = pos_encoding[:, :, 1::2]

    # assert tf.is_tensor(pos_encoding), "Output is not a tensor"
    assert pos_encoding.shape == (1, position, d_model), f"Wrong shape. We expected: (1, {position}, {d_model})"

    ones = sin_part ** 2 + cos_part ** 2
    assert np.allclose(ones,
                       np.ones((1, position, d_model // 2))), "Sum of square pairs must be 1 = sin(a)**2 + cos(a)**2"

    angs = np.arctan(sin_part / cos_part)
    angs[angs < 0] += np.pi
    angs[sin_part < 0] += np.pi
    angs = angs % (2 * np.pi)

    pos_m = np.arange(position)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]

    trueAngs = get_angles(pos_m, dims, d_model)[:, 0::2] % (2 * np.pi)

    assert np.allclose(angs[0], trueAngs), "Did you apply sin and cos to even and odd parts respectively?"

    print("\033[92mAll tests passed")

# positional_encoding_test(positional_encoding)

def create_padding_mask(seq):
    """
    Creates a matrix mask for the padding cells

    Arguments:
        seq -- (n, m) matrix

    Returns:
        mask -- (n, 1, 1, m) binary tensor
    """
    seq = torch.eq(seq, torch.tensor(0.)).float()  # tf.cast(tf.math.equal(seq, 0), tf.float32)
    return seq[:, None, None, :]  # seq[:, torch., tf.newaxis, :]

# x = torch.tensor([[7., 6., 0., 0., 1.], [1., 2., 3., 0., 0.], [0., 0., 0., 4., 5.]])
# print(create_padding_mask(x))


def create_look_ahead_mask(size):
    """
    Returns an upper triangular matrix filled with ones

    Arguments:
        size -- matrix size

    Returns:
        mask -- (size, size) tensor
    """
    mask = torch.tril(torch.ones((size, size)),
                      diagonal=0)  # https://blog.csdn.net/weixin_40548136/article/details/118698301
    # #tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask

x = torch.randint(0, 3, (3, 3))
temp = create_look_ahead_mask(x.shape[1])
print(temp)

def scaled_dot_product_attention(q, k, v, mask, dropout=None):
    """
    Calculate the attention weights.
      q, k, v must have matching leading dimensions.
      k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
      The mask has different shapes depending on its type(padding or look ahead)
      but it must be broadcastable for addition.

    Arguments:
        q -- query shape == (..., seq_len_q, depth)
        k -- key shape == (..., seq_len_k, depth)
        v -- value shape == (..., seq_len_v, depth_v)
        mask: Float tensor with shape broadcastable
              to (..., seq_len_q, seq_len_k). Defaults to None.

    Returns:
        output -- attention_weights
    """
    matmul_qk = torch.matmul(q, torch.transpose(k, -1, -2))
    # dk = torch.reshape(k.shape[0], -1)
    scaled_attention_logits = matmul_qk / torch.sqrt(torch.tensor(k.shape[-1], dtype=torch.float32))
    if mask is not None:
        scaled_attention_logits = scaled_attention_logits.masked_fill(mask==0, -1e9)
        #scaled_attention_logits += (mask * -1e9)
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)
    if dropout is not None:
        attention_weights = dropout(attention_weights)
    output = torch.matmul(attention_weights, v)
    return output, attention_weights


def scaled_dot_product_attention_test(target):
    q = np.array([[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]).astype(np.float32)
    k = np.array([[1, 1, 0, 1], [1, 0, 1, 1], [0, 1, 1, 0], [0, 0, 0, 1]]).astype(np.float32)
    v = np.array([[0, 0], [1, 0], [1, 0], [1, 1]]).astype(np.float32)

    attention, weights = target(torch.tensor(q, dtype=torch.float32), torch.tensor(k, dtype=torch.float32), torch.tensor(v, dtype=torch.float32), None)
    # assert tf.is_tensor(weights), "Weights must be a tensor"
    assert weights.shape == (
    q.shape[0], k.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {k.shape[1]})"
    assert np.allclose(weights, [[0.2589478, 0.42693272, 0.15705977, 0.15705977],
                                 [0.2772748, 0.2772748, 0.2772748, 0.16817567],
                                 [0.33620113, 0.33620113, 0.12368149, 0.2039163]])

    # assert tf.is_tensor(attention), "Output must be a tensor"
    assert attention.shape == (
    q.shape[0], v.shape[1]), f"Wrong shape. We expected ({q.shape[0]}, {v.shape[1]})"
    assert np.allclose(attention, [[0.74105227, 0.15705977],
                                   [0.7227253, 0.16817567],
                                   [0.6637989, 0.2039163]])

    mask = np.array([[0, 0, 1, 0], [0, 0, 1, 0], [0, 0, 1, 0]])
    attention, weights = target(torch.tensor(q, dtype=torch.float32), torch.tensor(k, dtype=torch.float32), torch.tensor(v, dtype=torch.float32), torch.tensor(mask, dtype=torch.float32))

    assert np.allclose(weights, [[0.30719590187072754, 0.5064803957939148, 0.0, 0.18632373213768005],
                                 [0.3836517333984375, 0.3836517333984375, 0.0, 0.2326965481042862],
                                 [0.3836517333984375, 0.3836517333984375, 0.0,
                                  0.2326965481042862]]), "Wrong masked weights"
    assert np.allclose(attention, [[0.6928040981292725, 0.18632373213768005],
                                   [0.6163482666015625, 0.2326965481042862],
                                   [0.6163482666015625, 0.2326965481042862]]), "Wrong masked attention"

    print("\033[92mAll tests passed")


# scaled_dot_product_attention_test(scaled_dot_product_attention)



def FullyConnected(embedding_dim, fully_connected_dim):
    return torch.nn.Sequential(
        torch.nn.LazyLinear(fully_connected_dim),  # (batch_size, seq_len, dff)
        torch.nn.ReLU(),
        torch.nn.LazyLinear(embedding_dim)  # (batch_size, seq_len, d_model)
    )


class MultiheadAttention(torch.nn.Module):
    def __init__(self, d_model=512, n_head=8):
        super(MultiheadAttention, self).__init__()
        self.linear_q = torch.nn.Linear(d_model, d_model)
        self.linear_k = torch.nn.Linear(d_model, d_model)
        self.linear_v = torch.nn.Linear(d_model, d_model)
        self.linear_out = torch.nn.Linear(d_model, d_model)
        assert (d_model % n_head == 0)
        self.dk = d_model // n_head
        self.d_model = d_model
        self.n_head = n_head

    def forward(self, q, k, v, mask=None):
        # if mask is not None:
        #     # 多头注意力机制的线性变换层是4维，是把query[batch, frame_num, d_model]变成[batch, -1, head, d_k]
        #     # 再1，2维交换变成[batch, head, -1, d_k], 所以mask要在第一维添加一维，与后面的self attention计算维度一样
        #     mask = mask.unsqueeze(1)

        n_batch = q.shape[0]
        # 多头需要对这个 X 切分成多头
        query = self.linear_q(q).view(n_batch, -1, self.n_head, self.dk).transpose(1, 2)  # [b, 8, 32, 64]
        key = self.linear_k(k).view(n_batch, self.n_head, -1, self.dk)  # [b, 8, 32, 64]
        value = self.linear_v(v).view(n_batch, self.n_head, -1, self.dk)  # [b, 8, 32, 64]
        output, attention_weights = scaled_dot_product_attention(query, key, value, mask)
        output = output.view(n_batch, -1, self.d_model)  # 相当于把多头联合到一起
        return self.linear_out(output), attention_weights


class EncoderLayer(torch.nn.Module):
    def __init__(self, d_model=512, n_head=8, fully_connected=2048, dropout_rate=0.1):
        super(EncoderLayer, self).__init__()
        self.d_model = d_model
        self.fully_connected = fully_connected
        self.n_head = n_head
        self.multihead = MultiheadAttention(d_model, n_head)
        self.layernormal1 = torch.nn.LayerNorm(d_model)
        self.feedforward = FullyConnected(d_model, fully_connected)
        self.layernoraml2 = torch.nn.LayerNorm(d_model)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)

    def forward(self, X, mask):
        # input = X + positional_encoding(X)
        x, _ = self.multihead(X, X, X, mask)
        x = self.dropout1(x)
        xinput = self.layernormal1(x + X)

        x1 = self.feedforward(xinput)
        x1 = self.dropout2(x1)
        x = self.layernoraml2(xinput + x1)
        return x


def EncoderLayer_test(target):
    q = np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32)
    encoder_layer1 = EncoderLayer(4, 2, 8)
    torch.random.seed()
    encoded = encoder_layer1(torch.tensor(q, dtype=torch.float32),  torch.tensor(np.array([[1, 0, 1]]), dtype=torch.float32), None)

    # assert tf.is_tensor(encoded), "Wrong type. Output must be a tensor"
    assert encoded.shape == (
    1, q.shape[1], q.shape[2]), f"Wrong shape. We expected ((1, {q.shape[1]}, {q.shape[2]}))"

    # assert np.allclose(encoded.numpy(),
    #                    [[-0.5214877, -1.001476, -0.12321664, 1.6461804],
    #                     [-1.3114998, 1.2167752, -0.5830886, 0.6778133],
    #                     [0.25485858, 0.3776546, -1.6564771, 1.023964]], ), "Wrong values"

    print("\033[92mAll tests passed")


# EncoderLayer_test(EncoderLayer)

class Encoder(torch.nn.Module):
    def __init__(self, input_vocab_size, maximum_position_encoding, num_layers=6, embedding_dim=512, num_heads=8,
                 fully_connected_dim=2048,
                 dropout_rate=0.1, layernorm_eps=1e-6):
        super(Encoder, self).__init__()
        self.input_vocab_size = input_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.fully_connected_dim = fully_connected_dim
        self.dropout_rate = dropout_rate
        self.layernorm_eps = layernorm_eps
        self.embedding = torch.nn.Embedding(input_vocab_size, embedding_dim)
        self.layers = []
        for i in range(num_layers):
            self.layers.append(EncoderLayer(embedding_dim, num_heads, fully_connected_dim, dropout_rate))
        self.dropout = torch.nn.Dropout(dropout_rate)

    def forward(self, X, mask):
        seq_len = X.shape[1]
        x = self.embedding(X)
        x *= torch.sqrt(torch.tensor(self.embedding_dim, dtype=torch.float32))
        x += positional_encoding(self.maximum_position_encoding, self.embedding_dim)[:, :seq_len, :]
        x = self.dropout(x)
        for i in range(self.num_layers):
            x = self.layers[i](x, mask)
        return x


def Encoder_test(target):
    # tf.random.set_seed(10)

    embedding_dim = 4

    encoderq = target(num_layers=2,
                      embedding_dim=embedding_dim,
                      num_heads=2,
                      fully_connected_dim=8,
                      input_vocab_size=32,
                      maximum_position_encoding=5)

    x = torch.tensor(np.array([[2, 1, 3], [1, 2, 0]]), dtype=torch.int)

    encoderq_output = encoderq(x,  None)

    # assert tf.is_tensor(encoderq_output), "Wrong type. Output must be a tensor"
    assert encoderq_output.shape == (
    x.shape[0], x.shape[1], embedding_dim), f"Wrong shape. We expected ({x.shape[0]}, {x.shape[1]}, {embedding_dim})"
    # assert np.allclose(encoderq_output.numpy(),
    #                    [[[-0.40172306, 0.11519244, -1.2322885, 1.5188192],
    #                      [0.4017268, 0.33922842, -1.6836855, 0.9427304],
    #                      [0.4685002, -1.6252842, 0.09368491, 1.063099]],
    #                     [[-0.3489219, 0.31335592, -1.3568854, 1.3924513],
    #                      [-0.08761203, -0.1680029, -1.2742313, 1.5298463],
    #                      [0.2627198, -1.6140151, 0.2212624, 1.130033]]]), "Wrong values"

    print("\033[92mAll tests passed")


# Encoder_test(Encoder)


class DecoderLayer(torch.nn.Module):
    def __init__(self, embedding_dim=512, num_heads=8, fully_connected_dim=2048, dropout_rate=0.1, layernorm_eps=1e-6):
        super(DecoderLayer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.fully_connected_dim = fully_connected_dim
        self.dropout_rate = dropout_rate
        self.layernorm_eps = layernorm_eps
        self.multihead1 = MultiheadAttention(embedding_dim, num_heads)
        self.multihead2 = MultiheadAttention(embedding_dim, num_heads)
        self.feedback = FullyConnected(self.embedding_dim, self.fully_connected_dim)
        self.layernorm1 = torch.nn.LayerNorm(self.embedding_dim, layernorm_eps)
        self.layernorm2 = torch.nn.LayerNorm(self.embedding_dim, layernorm_eps)
        self.layernorm3 = torch.nn.LayerNorm(self.embedding_dim, layernorm_eps)
        self.dropout1 = torch.nn.Dropout(dropout_rate)
        self.dropout2 = torch.nn.Dropout(dropout_rate)
        self.dropout3 = torch.nn.Dropout(dropout_rate)

    def forward(self, X, enc_output, look_ahead_mask, padding_mask):
        x, attn_weights_block1 = self.multihead1(X, X, X, look_ahead_mask)
        x = self.dropout1(x)
        x1 = self.layernorm1(x + X)

        # k, v = enc_output
        x2, attn_weights_block2 = self.multihead2(x1, enc_output, enc_output, padding_mask)
        x2 = self.dropout2(x2)
        x2out = self.layernorm2(x1 + x2)

        x3 = self.feedback(x2out)
        x3 = self.dropout3(x3)
        x3out = self.layernorm3(x3 + x2out)

        return x3out, attn_weights_block1, attn_weights_block2


def DecoderLayer_test(target):
    num_heads = 2
    torch.random.seed()

    decoderLayerq = target(
        embedding_dim=4,
        num_heads=num_heads,
        fully_connected_dim=32,
        dropout_rate=0.1,
        layernorm_eps=1e-6)

    encoderq_output = torch.tensor([[[-0.40172306, 0.11519244, -1.2322885, 1.5188192],
                                    [0.4017268, 0.33922842, -1.6836855, 0.9427304],
                                    [0.4685002, -1.6252842, 0.09368491, 1.063099]]], dtype=torch.float32)

    q = torch.tensor(np.array([[[1, 0, 1, 1], [0, 1, 1, 1], [1, 0, 0, 1]]]).astype(np.float32))

    look_ahead_mask = torch.tensor([[1., 0., 0.],
                                   [1., 1., 0.],
                                   [1., 1., 1.]], dtype=torch.float32)

    padding_mask = None
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output, look_ahead_mask, padding_mask)

    # assert tf.is_tensor(attn_w_b1), "Wrong type for attn_w_b1. Output must be a tensor"
    # assert tf.is_tensor(attn_w_b2), "Wrong type for attn_w_b2. Output must be a tensor"
    # assert tf.is_tensor(out), "Wrong type for out. Output must be a tensor"

    shape1 = (q.shape[0], num_heads, q.shape[1], q.shape[1])
    assert attn_w_b1.shape == shape1, f"Wrong shape. We expected {shape1}"
    assert attn_w_b2.shape == shape1, f"Wrong shape. We expected {shape1}"
    assert out.shape == q.shape, f"Wrong shape. We expected {q.shape}"

    # assert np.allclose(attn_w_b1.detach().numpy()[0, 0, 1], [0.5271505, 0.47284946, 0.],
    #                    atol=1e-2), "Wrong values in attn_w_b1. Check the call to self.mha1"
    # assert np.allclose(attn_w_b2[0, 0, 1],
    #                    [0.33365652, 0.32598493, 0.34035856]), "Wrong values in attn_w_b2. Check the call to self.mha2"
    # assert np.allclose(out[0, 0], [0.04726627, -1.6235218, 1.0327158, 0.54353976]), "Wrong values in out"

    # Now let's try a example with padding mask
    padding_mask = torch.tensor(np.array([[0, 0, 1]]), dtype=torch.float32)
    out, attn_w_b1, attn_w_b2 = decoderLayerq(q, encoderq_output,  look_ahead_mask, padding_mask)

    # assert np.allclose(out[0, 0], [-0.34323323, -1.4689083, 1.1092525,
    #                                0.7028891]), "Wrong values in out when we mask the last word. Are you passing the padding_mask to the inner functions?"

    print("\033[92mAll tests passed")


# DecoderLayer_test(DecoderLayer)


class Decoder(torch.nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim=2048, target_vocab_size=30,
               maximum_position_encoding=5,  dropout_rate=0.1, layernorm_eps=1e-6):
        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.fully_connected_dim = fully_connected_dim
        self.len_verb_size = target_vocab_size
        self.maximum_position_encoding = maximum_position_encoding
        self.embedding = torch.nn.Embedding(target_vocab_size, embedding_dim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layers = []
        for i in range(self.num_layers):
            self.layers.append(DecoderLayer(embedding_dim, num_heads, self.fully_connected_dim, layernorm_eps))
    def forward(self, X, enc_output, look_ahead_mask, padding_mask):
        seq_len = X.shape[1]
        attention_weights = {}
        x = self.embedding(X)
        encoding = positional_encoding(self.maximum_position_encoding, self.embedding_dim)
        x = x + encoding[:, :seq_len, :]
        for i in range(self.num_layers):
            x, block1, block2 = self.layers[i](x, enc_output, look_ahead_mask, padding_mask)
            # update attention_weights dictionary with the attention weights of block 1 and block 2
            attention_weights['decoder_layer{}_block1_self_att'.format(i + 1)] = block1
            attention_weights['decoder_layer{}_block2_decenc_att'.format(i + 1)] = block2
        return x, attention_weights


def Decoder_test(target):
    # tf.random.set_seed(10)

    num_layers = 7
    embedding_dim = 4
    num_heads = 2
    fully_connected_dim = 8
    target_vocab_size = 33
    maximum_position_encoding = 6

    x = torch.tensor(np.array([[3, 2, 1], [2, 1, 0]]), dtype=torch.int)

    encoderq_output = torch.tensor([[[-0.40172306, 0.11519244, -1.2322885, 1.5188192],
                                    [0.4017268, 0.33922842, -1.6836855, 0.9427304],
                                    [0.4685002, -1.6252842, 0.09368491, 1.063099]],
                                   [[-0.3489219, 0.31335592, -1.3568854, 1.3924513],
                                    [-0.08761203, -0.1680029, -1.2742313, 1.5298463],
                                    [0.2627198, -1.6140151, 0.2212624, 1.130033]]], dtype=torch.float32)

    look_ahead_mask = torch.tensor([[1., 0., 0.],
                                   [1., 1., 0.],
                                   [1., 1., 1.]], dtype=torch.float32)

    decoderk = Decoder(num_layers,
                       embedding_dim,
                       num_heads,
                       fully_connected_dim,
                       target_vocab_size,
                       maximum_position_encoding)
    outd, att_weights = decoderk(x, encoderq_output, look_ahead_mask, None)

    # assert tf.is_tensor(outd), "Wrong type for outd. It must be a dict"
    # assert np.allclose(tf.shape(outd),
    #                    tf.shape(encoderq_output)), f"Wrong shape. We expected {tf.shape(encoderq_output)}"
    print(outd[1, 1])
    # assert np.allclose(outd.detach().numpy()[1, 1], [-0.2715261, -0.5606001, -0.861783, 1.69390933]), "Wrong values in outd"

    keys = list(att_weights.keys())
    assert type(att_weights) == dict, "Wrong type for att_weights[0]. Output must be a tensor"
    assert len(
        keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2 * num_layers}"
    # assert tf.is_tensor(att_weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"
    shape1 = (x.shape[0], num_heads, x.shape[1], x.shape[1])
    assert att_weights[keys[1]].shape == shape1, f"Wrong shape. We expected {shape1}"
    # assert np.allclose(att_weights[keys[0]][0, 0, 1],
    #                    [0.52145624, 0.47854376, 0.]), f"Wrong values in att_weights[{keys[0]}]"

    print("\033[92mAll tests passed")


# Decoder_test(Decoder)

class Transformer(torch.nn.Module):
    def __init__(self, num_layers, embedding_dim, num_heads, fully_connected_dim, input_vocab_size,
               target_vocab_size, max_positional_encoding_input,
               max_positional_encoding_target, dropout_rate=0.1, layernorm_eps=1e-6):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_vocab_size, max_positional_encoding_input, num_layers, embedding_dim, num_heads, fully_connected_dim,
                               dropout_rate, layernorm_eps)
        self.decoder = Decoder(num_layers, embedding_dim, num_heads,  fully_connected_dim, target_vocab_size, max_positional_encoding_input,
                               dropout_rate, layernorm_eps)
        self.linear = torch.nn.LazyLinear(target_vocab_size)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, inp, tar, enc_padding_mask, look_ahead_mask, dec_padding_mask):
        """
        Forward pass for the entire Transformer
        Arguments:
            inp -- Tensor of shape (batch_size, input_seq_len, fully_connected_dim)
            tar -- Tensor of shape (batch_size, target_seq_len, fully_connected_dim)
            training -- Boolean, set to true to activate
                        the training mode for dropout layers
            enc_padding_mask -- Boolean mask to ensure that the padding is not
                    treated as part of the input
            look_ahead_mask -- Boolean mask for the target_input
            padding_mask -- Boolean mask for the second multihead attention layer
        Returns:
            final_output -- Describe me
            attention_weights - Dictionary of tensors containing all the attention weights for the decoder
                                each of shape Tensor of shape (batch_size, num_heads, target_seq_len, input_seq_len)

        """
        x = self.encoder(inp, enc_padding_mask)
        x, attention_weights = self.decoder(tar, x, look_ahead_mask, dec_padding_mask)
        x = self.linear(x)
        x = self.softmax(x)
        return x, attention_weights


def Transformer_test(target):
    torch.random.manual_seed(10)

    num_layers = 6
    embedding_dim = 4
    num_heads = 4
    fully_connected_dim = 8
    input_vocab_size = 30
    target_vocab_size = 35
    max_positional_encoding_input = 5
    max_positional_encoding_target = 6

    trans = Transformer(num_layers,
                        embedding_dim,
                        num_heads,
                        fully_connected_dim,
                        input_vocab_size,
                        target_vocab_size,
                        max_positional_encoding_input,
                        max_positional_encoding_target)
    # 0 is the padding value
    sentence_lang_a = torch.tensor(np.array([[2, 1, 4, 3, 0]]), dtype=torch.int)
    sentence_lang_b = torch.tensor(np.array([[3, 2, 1, 0, 0]]), dtype=torch.int)

    enc_padding_mask = torch.tensor(np.array([[0, 0, 0, 0, 1]]), dtype=torch.int)
    dec_padding_mask = torch.tensor(np.array([[0, 0, 0, 1, 1]]), dtype=torch.int)

    look_ahead_mask = create_look_ahead_mask(sentence_lang_a.shape[1])

    translation, weights = trans(
        sentence_lang_a,
        sentence_lang_b,
        enc_padding_mask,
        look_ahead_mask,
        dec_padding_mask
    )

    # assert tf.is_tensor(translation), "Wrong type for translation. Output must be a tensor"
    shape1 = (sentence_lang_a.shape[0], max_positional_encoding_input, target_vocab_size)
    assert translation.shape == shape1, f"Wrong shape. We expected {shape1}"

    print(translation[0, 0, 0:8])
    # assert np.allclose(translation.detach().numpy()[0, 0, 0:8],
    #                    [[0.02616475, 0.02074359, 0.01675757,
    #                      0.025527, 0.04473696, 0.02171909,
    #                      0.01542725, 0.03658631]]), "Wrong values in outd"

    keys = list(weights.keys())
    assert type(weights) == dict, "Wrong type for weights. It must be a dict"
    assert len(
        keys) == 2 * num_layers, f"Wrong length for attention weights. It must be 2 x num_layers = {2 * num_layers}"
    # assert tf.is_tensor(weights[keys[0]]), f"Wrong type for att_weights[{keys[0]}]. Output must be a tensor"

    shape1 = (sentence_lang_a.shape[0], num_heads, sentence_lang_a.shape[1], sentence_lang_a.shape[1])
    assert weights[keys[1]].shape == shape1, f"Wrong shape. We expected {shape1}"
    # assert np.allclose(weights[keys[0]].detach().numpy()[0, 0, 1],
    #                    [0.4992985, 0.5007015, 0., 0., 0.]), f"Wrong values in weights[{keys[0]}]"

    print(translation)

    print("\033[92mAll tests passed")


Transformer_test(Transformer)