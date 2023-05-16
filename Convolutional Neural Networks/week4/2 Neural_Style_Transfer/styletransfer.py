# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Conv2D, ZeroPadding2D, Activation, Input, concatenate
# from tensorflow.keras.models import Model
# from tensorflow.keras.layers import BatchNormalization
# from tensorflow.keras.layers import MaxPooling2D, AveragePooling2D
# from tensorflow.keras.layers import Concatenate
# from tensorflow.keras.layers import Lambda, Flatten, Dense
# from tensorflow.keras.initializers import glorot_uniform
# from tensorflow.keras.layers import Layer
# from tensorflow.keras import backend as K
# K.set_image_data_format('channels_last')
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from torch.nn import Sequential, Conv2d, ZeroPad2d, Module, BatchNorm2d, MaxPool2d, AvgPool2d, Flatten
import torch
import os
import numpy as np
from numpy import genfromtxt
import pandas as pd
# import tensorflow as tf
import PIL


def triplet_loss(y_true, y_pred, alpha=0.2):
    """
    Implementation of the triplet loss as defined by formula (3)

    Arguments:
    y_true -- true labels, required when you define a loss in Keras, you don't need it in this function.
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)

    Returns:
    loss -- real number, value of the loss
    """
    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    pos_dist = torch.sum(torch.square(torch.subtract(anchor, positive)), dim=-1)
    # Step 1: Compute the (encoding) distance between the anchor and the positive
    neg_dist = torch.sum(torch.square(torch.subtract(anchor, positive)), dim=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = torch.maximum(torch.add(torch.subtract(pos_dist, neg_dist), alpha), torch.tensor(0))
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = torch.sum(basic_loss)
    return loss


def compute_content_cost(content_output, generated_output):
    """
    Computes the content cost

    Arguments:
    a_C -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image C
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing content of the image G

    Returns:
    J_content -- scalar that you compute using equation 1 above.
    """
    a_C = content_output[-1]
    a_G = generated_output[-1]

    ### START CODE HERE

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = list(a_G.size())
    # Reshape a_C and a_G (≈2 lines)
    a_C_unrolled = torch.reshape(a_C, shape=[m, n_H * n_W, n_C])  # Or tf.reshape(a_C, shape=[m, -1 , n_C])
    a_G_unrolled = torch.reshape(a_G, shape=[m, n_H * n_W, n_C])  # Or tf.reshape(a_G, shape=[m, -1 , n_C])

    # compute the cost with tensorflow (≈1 line)
    J_content = torch.sum(torch.square(a_C_unrolled - a_G_unrolled)) / (4.0 * n_H * n_W * n_C)
    return J_content


def gram_matrix(A):
    """
    Argument:
    A -- matrix of shape (n_C, n_H*n_W)

    Returns:
    GA -- Gram matrix of A, of shape (n_C, n_C)
    """
    ### START CODE HERE
    # (≈1 line)

    GA = torch.matmul(A, A.T)

    ### END CODE HERE

    return GA


def compute_layer_style_cost(a_S, a_G):
    """
    Arguments:
    a_S -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image S
    a_G -- tensor of dimension (1, n_H, n_W, n_C), hidden layer activations representing style of the image G

    Returns:
    J_style_layer -- tensor representing a scalar value, style cost defined above by equation (2)
    """
    ### START CODE HERE

    # Retrieve dimensions from a_G (≈1 line)
    m, n_H, n_W, n_C = list(a_G.size())
    # Reshape the images to have them of shape (n_C, n_H*n_W) (≈2 lines)
    a_S = torch.transpose(torch.reshape(a_S, shape=[-1, n_C]), 0, 1)
    # OR a_S = tf.transpose(tf.reshape(a_S, shape=[ n_H * n_W, n_C]))
    a_G = torch.transpose(torch.reshape(a_G, shape=[-1, n_C]), 0, 1)
    # Computing gram_matrices for both images S and G (≈2 lines)
    GS = gram_matrix(a_S)
    GG = gram_matrix(a_G)

    # Computing the loss (≈1 line)
    J_style_layer = torch.sum(torch.square(GS - GG)) / (4.0 * ((n_H * n_W * n_C) ** 2))

    ### END CODE HERE

    return J_style_layer


STYLE_LAYERS = [
    ('block1_conv1', 1.0),
    ('block2_conv1', 0.8),
    ('block3_conv1', 0.7),
    ('block4_conv1', 0.2),
    ('block5_conv1', 0.1)]

def compute_style_cost(style_image_output, generated_image_output, STYLE_LAYERS=STYLE_LAYERS):
    """
    Computes the overall style cost from several chosen layers

    Arguments:
    style_image_output -- our tensorflow model
    generated_image_output --
    STYLE_LAYERS -- A python list containing:
                        - the names of the layers we would like to extract style from
                        - a coefficient for each of them

    Returns:
    J_style -- tensor representing a scalar value, style cost defined above by equation (2)
    """

    # initialize the overall style cost
    J_style = 0

    # Set a_S to be the hidden layer activation from the layer we have selected.
    # The first element of the array contains the input layer image, which must not to be used.
    a_S = style_image_output[1:]

    # Set a_G to be the output of the choosen hidden layers.
    # The First element of the list contains the input layer image which must not to be used.
    a_G = generated_image_output[1:]
    for i, weight in zip(range(len(a_S)), STYLE_LAYERS):
        # Compute style_cost for the current layer
        J_style_layer = compute_layer_style_cost(a_S[i], a_G[i])

        # Add weight * J_style_layer of this layer to overall style cost
        J_style += weight[1] * J_style_layer

    return J_style


def total_cost(J_content, J_style, alpha=10, beta=40):
    """
    Computes the total cost function

    Arguments:
    J_content -- content cost coded above
    J_style -- style cost coded above
    alpha -- hyperparameter weighting the importance of the content cost
    beta -- hyperparameter weighting the importance of the style cost

    Returns:
    J -- total cost as defined by the formula above.
    """
    ### START CODE HERE

    # (≈1 line)
    J = alpha * J_content + beta * J_style

    ### START CODE HERE

    return J




class Gram(torch.nn.Module):
    def __init__(self):
        super(Gram, self).__init__()

    def forward(self, input):
        a, b, c, d = input.size()
        # 将特征图变换为 2 维向量
        feature = input.view(a * b, c * d)
        # 内积的计算方法其实就是特征图乘以它的逆
        gram = torch.mm(feature, feature.t())
        # 对得到的结果取平均值
        gram /= (a * b * c * d)
        return gram

class Content_Loss(torch.nn.Module):
    # 其中 target 表示 C ，input 表示 G，weight 表示 alpha 的平方
    def __init__(self, target, weight):
        super(Content_Loss, self).__init__()
        self.weight = weight
        # detach 可以理解为使 target 能够动态计算梯度
        # target 表示目标内容，即想变成的内容
        self.target = target.detach() * self.weight
        self.criterion = torch.nn.MSELoss()

    def forward(self, input):
        self.loss = self.criterion(input * self.weight, self.target)
        out = input.clone()
        return out

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss

class Style_Loss(torch.nn.Module):
    def __init__(self, target, weight):
        super(Style_Loss, self).__init__()
        # weight 和内容函数相似，表示的是权重 beta
        self.weight = weight
        # targer 表示图层目标。即新图像想要拥有的风格
        # 即保存目标风格
        self.target = target.detach() * self.weight
        self.gram = Gram()
        self.criterion = torch.nn.MSELoss()

    def forward(self, input):
        # 加权计算 input 的 Gram 矩阵
        G = self.gram(input) * self.weight
        # 计算真实的风格和想要得到的风格之间的风格损失
        self.loss = self.criterion(G, self.target)
        out = input.clone()
        return out
    # 向后传播

    def backward(self, retain_variabels=True):
        self.loss.backward(retain_graph=retain_variabels)
        return self.loss

from torchvision.models import vgg19
# model = vgg19(pretrained=True)
vgg = vgg19(pretrained=True).features
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
# 初始化一个 空的神经网络 model
model = torch.nn.Sequential()

def get_style_model_and_loss(style_img, content_img, cnn=vgg, style_weight=1000, content_weight=1,
                             content_layers=content_layers_default,
                             style_layers=style_layers_default):
    # 用列表来存上面6个损失函数
    content_loss_list = []
    style_loss_list = []

    # 风格提取函数
    gram = Gram()
    i = 1
    # 遍历 VGG19 ，找到其中我们需要的卷积层
    for layer in cnn:
        # 如果 layer 是  nn.Conv2d 对象，则返回 True
        # 否则返回 False
        if isinstance(layer, torch.nn.Conv2d):
            # 将该卷积层加入我们的模型中
            name = 'conv_' + str(i)
            model.add_module(name, layer)

            # 判断该卷积层是否用于计算内容损失
            if name in content_layers_default:
                # 这里是把目标放入模型中，得到该层的目标
                target = model(content_img)
                # 目标作为参数传入具体的损失类中，得到一个工具函数。
                # 该函数可以计算任何图片与目标的内容损失
                content_loss = Content_Loss(target, content_weight)
                model.add_module('content_loss_' + str(i), content_loss)
                content_loss_list.append(content_loss)

            # 和内容损失相似，不过增加了一步：提取风格
            if name in style_layers_default:
                target = model(style_img)
                target = gram(target)
                # 目标作为参数传入具体的损失类中，得到一个工具函数。
                # 该函数可以计算任何图片与目标的风格损失
                style_loss = Style_Loss(target, style_weight)
                model.add_module('style_loss_' + str(i), style_loss)
                style_loss_list.append(style_loss)

            i += 1
        # 对于池化层和 Relu 层我们直接添加即可
        if isinstance(layer, torch.nn.MaxPool2d):
            name = 'pool_' + str(i)
            model.add_module(name, layer)

        if isinstance(layer, torch.nn.ReLU):
            name = 'relu' + str(i)
            model.add_module(name, layer)
    # 综上：我们得到了：
    # 一个具体的神经网络模型，
    # 一个风格损失函数集合（其中包含了 5 个不同风格目标的损失函数）
    # 一个内容损失函数集合（这里只有一个，你也可以多定义几个）
    return model, style_loss_list, content_loss_list



def get_input_param_optimier(input_img):
    # 将input_img的值转为神经网络中的参数类型
    input_param = torch.nn.Parameter(input_img.data)
    # 告诉优化器，我们优化的是 input_img 而不是网络层的权重
    # 采用 LBFGS 优化器
    optimizer = torch.optim.LBFGS([input_param])
    return input_param, optimizer

img_size = 512
import PIL.Image as Image
import torchvision.transforms as transforms
def load_img(img_path):
    img = Image.open(img_path).convert('RGB')
    img = img.resize((img_size, img_size))
    img = transforms.ToTensor()(img)
    # 为img增加一个维度：1
    # 因为神经网络的输入为 4 维
    img = img.unsqueeze(0)
    return img



from torch.autograd import Variable
import matplotlib.pyplot as plt
# 判断环境是否支持GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
style_path = "./images/my_style.jpg"
content_path = "./images/my_content.jpg"
# 加载风格图片
style_img = load_img(style_path)
# 对img进行转换为 Variable 对象，使它能够动态计算梯度
style_img = Variable(style_img).to(device)
# 加载内容图片
content_img = load_img(content_path)
content_img = Variable(content_img).to(device)


model, style_loss_list, content_loss_list = get_style_model_and_loss(
    style_img, content_img)


# 传入的 input_img 是 G 中每个像素点的值，可以为一个随机图片
def run_style_transfer(content_img, style_img, input_img, num_epoches):
    print('Building the style transfer model..')
    # 指定所需要优化的参数，这里 input_param就是G中的每个像素点的值
    input_param, optimizer = get_input_param_optimier(input_img)

    print('Opimizing...')
    epoch = [0]
    while epoch[0] < num_epoches:
        # 这里我们自定义了总损失的计算方法
        def closure():
            input_param.data.clamp_(0, 1)  # 更新图像的数据
            # 将此时的 G 传入模型中，得到每一个网络层的输出
            model(input_param)
            style_score = 0
            content_score = 0
            # 清空之前的梯度
            optimizer.zero_grad()
            # 计算总损失，并得到各个损失的梯度
            for sl in style_loss_list:
                style_score += sl.backward()
            for cl in content_loss_list:
                content_score += cl.backward()

            epoch[0] += 1
            # 这里每迭代一次就进行一次输出
            # 你可以根据自身情况进行调节
            if epoch[0] % 1 == 0:
                print('run {}/80'.format(epoch))
                print('Style Loss: {:.4f} Content Loss: {:.4f}'.format(
                    style_score.data.item(), content_score.data.item()))
                print()

            return style_score + content_score
        # 更新 G
        optimizer.step(closure)
    # 返回训练完成的 G，此时的 G
    return input_param.data

input_img = content_img.clone()
# 进行模型训练，并且返回图片
out = run_style_transfer(content_img, style_img, input_img, num_epoches=80)
# 将图片转换成可 PIL 类型，便于展示
new_pic = transforms.ToPILImage()(out.cpu().squeeze(0))
print("训练完成")
plt.imshow(new_pic)