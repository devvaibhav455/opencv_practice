"""
Dev Vaibhav
Spring 2023 CS 5330
Project 5: Recognition using Deep Networks 
Task 2: Examine your network 
"""



# import statements
import sys
import torch
import matplotlib.pyplot as plt

import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
import os
from PIL import Image
import cv2

# Helpful to understand imports: https://redirect.cs.umbc.edu/courses/331/fall11/notes/python/python3.ppt.pdf
# Import classes and functions from helper.py
from helper import *


# Main function which instantiates an object of the MyNetwork class , loads the weight from the already trained model from tasks 1A to 1E. I renamed the model name manually and load model_1A_1E.
def main(argv):

    # main function code
    network = MyNetwork()
    print("Model is: ", network)
    network.load_state_dict(torch.load("./results/model_1A_1E.pth"))
    # Model is set in evaluation mode
    network.eval()
    
    # Access the weights of conv1 layer
    #Src: https://discuss.pytorch.org/t/access-weights-of-a-specific-module-in-nn-sequential/3627
    weights_conv1 = network.conv1.weight
    print("Weights shape is: ", weights_conv1.shape)
    print("Weights of 0th filter are: ", weights_conv1[0][0])

    # 2 A. Visualizing the first layer filters
    # Will get an error if with torch.no_grad() is not used
    # Error: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead
    with torch.no_grad():   
        fig = plt.figure()
        # Plot all the 10 filters of conv1
        for i in range(10):
            plt.subplot(3,4,i+1) #The indexes of subplot start from 1 and increase in a row by one
            plt.tight_layout()
            plt.imshow(weights_conv1[i][0], interpolation='none')
            plt.title("Filter: {}".format(i))
            #Ticks on empty list removes the axis labeling (0,1,2....)
            plt.xticks([])
            plt.yticks([])
        fig
        plt.show()

    
    
    # Debug trace to print out the requires_grad flag, and layer information
    # print("Printing the parameters")
    # for param, (name,layer) in zip(network.parameters(), network.named_modules()):
    #   print("Requires_grad: ", param.requires_grad, " | Name: ", name, " | Layer: ", layer)

    #If dataset is already downloaded, it is not downloaded again.
    # Contains data for the training set
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True)


    examples = enumerate(train_loader)
    # Loads first batch of train_loader into example_data
    batch_idx, (example_data, example_targets) = next(examples)
    print("Example data shape: ", example_data.shape)
    
    # Telling Pytorch not to calculate gradients
    with torch.no_grad():   
        img = example_data[0][0].detach().numpy()
    plt.imshow(img, cmap='gray', interpolation='none') #Visualizing the first training example

    # Visualizing the effect of 10 filters of the first layer on the first training example
    with torch.no_grad():   
        fig = plt.figure()
        for i in range(0,20,2):
            plt.subplot(5,4,i+1)
            plt.tight_layout()
            plt.imshow(weights_conv1[i//2][0], cmap='gray', interpolation='none')
            plt.title("Filter: {}".format(i//2))
            plt.xticks([])
            plt.yticks([])

            #Src: https://stackoverflow.com/questions/34097281/convert-a-tensor-to-numpy-array-in-tensorflow
            # https://stackoverflow.com/questions/63582590/why-do-we-call-detach-before-calling-numpy-on-a-pytorch-tensor
            # ddepth = -1 means that the output image will be of the same size as that of the input.
            img = cv2.filter2D(src=example_data[0][0].detach().numpy(), ddepth=-1, kernel=weights_conv1[i//2][0].detach().numpy())
            plt.subplot(5,4,i+2)
            plt.tight_layout()
            plt.imshow(img, cmap='gray', interpolation='none')
            plt.title("Filtered image: {}".format(i//2))
            #Ticks on empty list removes the axis labeling (0,1,2....)
            plt.xticks([])
            plt.yticks([])
        fig
        plt.show()

    return

if __name__ == "__main__":
    # A bunch of self understood bariables governing the training and testing process
    n_epochs = 3 #Number of epochs
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    random_seed = 1
    torch.backends.cudnn.enabled = False #cuDNN uses nondeterministic algorithms which can be disabled 
    torch.manual_seed(random_seed)

    main(sys.argv)
