from termcolor import colored
import torch
# from tensorflow.keras.layers import Input
# from tensorflow.keras.layers import Conv2D
# from tensorflow.keras.layers import MaxPooling2D
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.layers import Conv2DTranspose
# from tensorflow.keras.layers import concatenate
# from tensorflow.keras.layers import ZeroPadding2D
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import LSTM
# from tensorflow.keras.layers import RepeatVector




from torch.nn.modules import Conv2d, MaxPool2d, Dropout, ZeroPad2d, LSTM, ConvTranspose2d, Linear, ReplicationPad2d


# Compare the two inputs
def comparator(learner, instructor):
    for a, b in zip(learner, instructor):
        if tuple(a) != tuple(b):
            print(colored("Test failed", attrs=['bold']),
                  "\n Expected value \n\n", colored(f"{b}", "green"), 
                  "\n\n does not match the input value: \n\n", 
                  colored(f"{a}", "red"))
            raise AssertionError("Error in test") 
    print(colored("All tests passed!", "green"))

# extracts the description of a given model
def summary(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    result = []
    for layer in model.layers:
        descriptors = [layer.__class__.__name__, layer.output_shape, layer.count_params()]
        if (type(layer) == Conv2d):#Conv2D
            descriptors.append(layer.padding)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.kernel_initializer.__class__.__name__)
        if (type(layer) == MaxPool2d):#MaxPooling2D
            descriptors.append(layer.pool_size)
            descriptors.append(layer.strides)
            descriptors.append(layer.padding)
        if (type(layer) == Dropout):
            descriptors.append(layer.rate)
        if (type(layer) == ZeroPad2d):#ZeroPadding2D
            descriptors.append(layer.padding)
        if (type(layer) == Linear):#Dense
            descriptors.append(layer.activation.__name__)
        if (type(layer) == LSTM):
            descriptors.append(layer.input_shape)
            descriptors.append(layer.activation.__name__)
            descriptors.append(layer.return_sequences)
        if (type(layer) == RepeatVector):
            descriptors.append(layer.n)
        result.append(descriptors)
    return result
