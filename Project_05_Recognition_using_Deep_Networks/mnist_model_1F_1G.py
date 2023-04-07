"""
Dev Vaibhav
Spring 2023 CS 5330
Project 5: Recognition using Deep Networks
Tasks 1F to 1G
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

# Helpful to understand imports: https://redirect.cs.umbc.edu/courses/331/fall11/notes/python/python3.ppt.pdf
# Import classes and functions from helper.py
from helper import *

# Main function which instantiates an object of the MyNetwork class , loads the weight from the already trained model from tasks 1A to 1E. I renamed the model name manually and load model_1A_1E.
def main(argv):
    # main function code
    network = MyNetwork()
    network.load_state_dict(torch.load("./results/model_1A_1E.pth"))
    # Model is set in evaluation mode
    network.eval()

    #If dataset is already downloaded, it is not downloaded again.
    # Contains data for the validation set (train is set to False)
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_test, shuffle=True)
    
    
    # Debug trace to print the shape of NW input
    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    examples = enumerate(test_loader)
    # Loads first batch of test_loader into example_data
    batch_idx, (example_data, example_targets) = next(examples)

    # Telling Pytorch not to calculate gradients
    with torch.no_grad():
        output = network(example_data)

    torch.set_printoptions(sci_mode=False)
    torch.set_printoptions(precision=2)
    for i in range(10):
        print("\nExample ", i)
        print("Output values: ", output[i])
        print("Index of maximum value: ", torch.argmax(output[i]))
        print("Correct label of the digit: ", output.data.max(1, keepdim=True)[1][i].item())

    # Plot the first 6 training examples from the test set against their ground truth
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            output.data.max(1, keepdim=True)[1][i].item()))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()

    ##############################################################################
    # TASK 1G: READ USER'S HANDWRITTEN IMAGES FROM A FOLDER and predict the digit
    ##############################################################################

    #Src: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder;
    # https://www.projectpro.io/recipes/convert-image-tensor-pytorch; 
    # https://www.analyticsvidhya.com/blog/2021/04/10-pytorch-transformations-you-need-to-know/
    
    # A set of operations to apply on the handwritten images to pass to the network
    convert_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), #Converts [0,255] to [0,1]
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(28, antialias=True),
        torchvision.transforms.RandomInvert(p=1.0),
        ThresholdTransform(thr_255=180),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]
    )
    # Initializing the batch with zeros
    batch = torch.zeros(10, 1,28,28)#,dtype=float)

    # Loading all the images inside the folder "handwritten_digits"
    my_folder = "./handwritten_digits/"
    i = 0
    for filename in os.listdir(my_folder):
        img = Image.open(os.path.join(my_folder,filename))
        img = convert_tensor(img) #img is 0/ 1 here after binary
        # print("img is:" , img)
        batch[i] = img
        i = i+1
    
    print("Batch shape: ", batch.shape)
    
    # Get the NW output on handwritten digits
    with torch.no_grad():
        handwritten_output = network(batch)

    fig = plt.figure()
    for i in range(10):
        plt.subplot(4,3,i+1)
        plt.tight_layout()
        plt.imshow(batch[i][0], cmap='gray', interpolation='none')
        plt.title("Prediction: {}".format(
            handwritten_output.data.max(1, keepdim=True)[1][i].item()))
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
