"""
Dev Vaibhav
Spring 2023 CS 5330
Project 5: Recognition using Deep Networks
Tasks 1A to 1E
"""

# import statements
import sys
import torch
import matplotlib.pyplot as plt

import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
# Helpful to understand imports: https://redirect.cs.umbc.edu/courses/331/fall11/notes/python/python3.ppt.pdf
# Import classes and functions from helper.py
from helper import *



# Main function which instantiates an object of the MyNetwork class , loads an optimizer, trains, tests the network and saves it to disk. System arguments are not used for this script
def main(argv):
    # handle any command line arguments in argv

    # main function code
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    #If dataset is already downloaded (in files folder), it is not downloaded again.
    # Contains data for the training set
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST('./files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True)

    # Contains data for the validation set
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

    # Plot the first 6 training examples from the train set against their ground truth
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()

    # Arrays to keep track of various losses
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]

    # Testing on validation set is done before starting the training process to evaluate the model with randomly initialized parameters which comes out to be "Test set: Avg. loss: 2.3316, Accuracy: 1137/10000 (11%)"
    test_network(network, test_loader, test_losses)
    # Train the network for the number of epochs and test it again as soon as testing on one epoch is complete
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, train_loader, optimizer, train_losses, train_counter)
        test_network(network, test_loader, test_losses)
    
    # Plot the training and validation loss
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
    plt.show()
    return

if __name__ == "__main__":
    
    # A bunch of self understood bariables governing the training and testing process
    n_epochs = 5 #Number of epochs
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5

    random_seed = 1
    torch.backends.cudnn.enabled = False #cuDNN uses nondeterministic algorithms which can be disabled 
    torch.manual_seed(random_seed)

    main(sys.argv)
