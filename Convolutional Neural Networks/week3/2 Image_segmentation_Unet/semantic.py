import numpy as np
from torch.nn import Conv2d, MaxPool2d, Dropout, ConvTranspose2d, ReLU,  Module
from torchvision.transforms import transforms, Compose
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from torch.utils.data import Dataset, DataLoader
import torch
import os
import imageio
from PIL import Image
import matplotlib.pyplot as plt

path = ''
image_path = os.path.join(path, './data/CameraRGB/')
mask_path = os.path.join(path, './data/CameraMask/')
image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)
image_list = [image_path + i for i in image_list]
mask_list = [mask_path + i for i in mask_list]


def test_show():
    N = 3
    img = imageio.imread(image_list[N])
    mask = imageio.imread(mask_list[N])
    # mask = np.array([max(mask[i, j]) for i in range(mask.shape[0]) for j in range(mask.shape[1])]).reshape(img.shape[0], img.shape[1])

    fig, arr = plt.subplots(1, 2, figsize=(14, 10))
    arr[0].imshow(img)
    arr[0].set_title('Image')
    arr[1].imshow(mask[:, :, 0])
    arr[1].set_title('Segmentation')


class ImageDataSet(Dataset):
    def __init__(self, data, label):
        self.data = data
        self.label = label
        self.transform = Compose([transforms.Resize((96, 128)), transforms.ToTensor()])

    def __getitem__(self, index):
        imagepath = self.data[index]
        labelpath = self.label[index]
        image = Image.Open(imagepath).Convert("RGB")
        imagedata = self.transform(image)
        label = Image.Open(labelpath).Convert("RGB")
        labeldata = self.transform(label)
        return imagedata, labeldata

    def __len__(self):
        return len(self.data)


class conv_block(Module):
    """
       Convolutional downsampling block

       Arguments:
           inputs -- Input tensor
           n_filters -- Number of filters for the convolutional layers
           dropout_prob -- Dropout probability
           max_pooling -- Use MaxPooling2D to reduce the spatial dimensions of the output volume
       Returns:
           next_layer, skip_connection --  Next layer and skip connection outputs
       """

    def __init__(self, inchannels, n_filters=32, dropout_prob=0., max_pooling=True):
        super(conv_block, self).__init__()
        self.conv1 = Conv2d(inchannels, n_filters, padding="same")
        self.dropout_prob = dropout_prob
        self.max_pooling = max_pooling
        self.drop = Dropout(dropout_prob)
        self.maxpool = MaxPool2d((2, 2), stride=2)
        self.relu = ReLU()

    def forward(self, X):
        conv = self.conv1(X)
        conv = self.relu(conv)
        conv = self.conv1(conv)
        conv = self.relu(conv)
        if self.dropout_prob > 0:
            conv = self.drop(conv)
        if self.max_pooling:
            next_layer = self.maxpool(conv)
        else:
            next_layer = conv
        skip_connection = conv
        return next_layer, skip_connection

class upsampling_block(Module):
    """
        Convolutional upsampling block

        Arguments:
            expansive_input -- Input tensor from previous layer
            contractive_input -- Input tensor from previous skip layer
            n_filters -- Number of filters for the convolutional layers
        Returns:
            conv -- Tensor output
        """
    def __init__(self, inchannels, n_filters=32):
        super(upsampling_block, self).__init__()
        self.convTranspose = ConvTranspose2d(inchannels, n_filters, (3, 3), (2, 2))
        self.conv1 = Conv2d(n_filters, n_filters, (3, 3), padding="same")
        self.relu = ReLU()
    def forward(self, X):
        expansive_input, contractive_input = X
        up = self.convTranspose(expansive_input)
        merge = torch.concat([up, contractive_input], dim=3)
        conv = self.conv(merge)
        conv = self.relu(conv)
        conv = self.conv(conv)
        conv = self.relu(conv)
        return conv


class unet_model(Module):
    def __init__(self, inchannels, n_filters=32, n_classes=23):
        self.n_filters = n_filters
        self.n_classes = n_classes
        # Add a conv_block with the inputs of the unet_ model and n_filters
        self.conv_block1 = conv_block(inchannels, n_filters*1)
        # Chain the first element of the output of each block to be the input of the next conv_block.
        # Double the number of filters at each new step
        self.conv_block2 = conv_block(n_filters, n_filters * 2)

        self.conv_block3 = conv_block(n_filters*2, n_filters*4)
        # Include a dropout of 0.3 for this layer
        self.conv_block4 = conv_block(n_filters*4, n_filters*8, dropout_prob=0.3)
        # Include a dropout of 0.3 for this layer, and avoid the max_pooling layer
        self.conv_block5 = conv_block(n_filters*8, n_filters*16, dropout_prob=0.3, max_pooling=False)

        # Expanding Path (decoding)
        # Add the first upsampling_block.
        # From here,at each step, use half the number of filters of the previous block
        # Use the cblock5[0] as expansive_input and cblock4[1] as contractive_input and n_filters * 8
        ### START CODE HERE
        self.ublock1 = upsampling_block(n_filters*16, n_filters*8)
        # Chain the output of the previous block as expansive_input and the corresponding contractive block output.
        # Note that you must use the second element of the contractive block i.e before the maxpooling layer.
        self.ublock2 = upsampling_block(n_filters*8, n_filters*4)
        self.ublock3 = upsampling_block(n_filters*4, n_filters*2)
        self.ublock4 = upsampling_block(n_filters*2, n_filters)

        self.conv9 = Conv2d(n_filters, n_filters, (3, 3), padding="same")
        self.fc = Conv2d(n_filters, n_classes, (1, 1), padding="same")
        self.relu = ReLU()

    def forward(self, X):
        conv_block1 = self.conv_block1(X)
        conv_block2 = self.conv_block2(conv_block1[0])
        conv_block3 = self.conv_block3(conv_block2[0])
        conv_block4 = self.conv_block4(conv_block3[0])
        conv_block5 = self.conv_block5(conv_block4[0])

        ubblock1  = self.ublock1((conv_block5[0], conv_block4[1]))
        ubblock2 = self.ublock2((ubblock1, conv_block3[1]))
        ubblock3 = self.ublock3((ubblock2, conv_block2[1]))
        ubblock4 = self.ublock4((ubblock3, conv_block1[1]))

        conv = self.conv9(ubblock4)
        conv = self.relu(conv)
        return self.fc(conv)


def loss_batch(model, loss_func, xb, yb, opt=None):
    z = model(xb)
    loss = loss_func(z, yb)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    return loss.item(), len(xb)

def fit():
    model = unet_model(3)
    dataset = ImageDataSet(image_list, mask_list)
    trainLoader = DataLoader(dataset=dataset, batch_size=16, shuffle=True)
    loss_func = torch.nn.CrossEntropyLoss()
    opt = torch.optim.Adam(model.parameters())
    epochs = 50
    print_cost = True
    trainloss = []
    for i in range(epochs):
        model.train()
        for datax, datay in trainLoader:
            opt.zero_grad()
            p = model(datax)
            loss = loss_func(p, datay)
            loss.back_ward()
            opt.step()
            nums = len(datay)#datay.numel()
            val_loss = np.sum(np.multiply(loss.item(), nums)) / np.sum(nums)
            if print_cost == True:
                print("val_loss after epoch %i: %f" % (i, val_loss))
                trainloss.append(val_loss)
        model.eval()

