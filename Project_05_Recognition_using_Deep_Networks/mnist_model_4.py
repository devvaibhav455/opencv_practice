"""
Dev Vaibhav
Spring 2023 CS 5330
Project 5: Recognition using Deep Networks 
Task 4
"""



#Src: https://nextjournal.com/gkoehler/pytorch-mnist
# import statements
import sys
import torch
import matplotlib.pyplot as plt

import torchvision
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from helper import *

# Main function which instantiates an object of the MyNetwork class, trains the network for 540 combinations and saves the result to a CSV file
def main(argv):
    # handle any command line arguments in argv

    # main function code
    network = MyNetwork()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

    #If dataset is already downloaded, it is not downloaded again.
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./fashion_mnist_files/', train=True, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_train, shuffle=True)

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.FashionMNIST('./fashion_mnist_files/', train=False, download=True,
                                    transform=torchvision.transforms.Compose([
                                    torchvision.transforms.ToTensor(),
                                    torchvision.transforms.Normalize(
                                        (0.1307,), (0.3081,))
                                    ])),
        batch_size=batch_size_test, shuffle=True)

    for X, y in test_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    examples = enumerate(test_loader)
    batch_idx, (example_data, example_targets) = next(examples)

    #See the initial 6 images in test set
    fig = plt.figure()
    for i in range(6):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig
    # plt.show()

    # Different losses (self explanatory)
    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    accuracies = [] #Accuracy for each epoch after network is tested on test set

    accuracies.append(test_network(network, test_loader, test_losses))
    
    # Train and test the network
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, train_loader, optimizer, train_losses, train_counter)
        accuracies.append(test_network(network, test_loader, test_losses))

    #WRITE THE RESULTS TO A FILE TO LATER USE
    # Opening a file
    file1 = open('network_output.txt', 'a')

    # print("Accuracy is: ", round(accuracies[-1].item(),2))
    
    s = str(batch_size_train) + ","\
    + str(dropout_rate) + ","\
    + str(learning_rate) + ","\
    + str(momentum) + ","\
    + str(round(train_losses[-1],4)) + ","\
    + str(round(test_losses[-1],4)) + ","\
    + str(n_epochs) + ","\
    + str(round(accuracies[-1].item(),2)) + "\n"

    # Writing the stats to file
    file1.write(s)

    # Closing file
    file1.close()
    
    # Plotting the output
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
    # plt.show()
    return

if __name__ == "__main__":
    
    # Variations of the network
    n_epoch_list = [3,4,5] #3
    batch_size_train_list = [16, 32, 64, 128] #4
    dropout_rate_list = [0.2,0.4, 0.5, 0.7, 0.8] #5
    learning_rate_list = [0.001, 0.01, 0.1] #3
    momentum_list = [0.4, 0.5, 0.6] #3

    #DEFAULT VALUES
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10
    dropout_rate = 0.5

    random_seed = 1
    torch.backends.cudnn.enabled = False #cuDNN uses nondeterministic algorithms which can be disabled 
    torch.manual_seed(random_seed)

    #WRITE THE RESULTS TO A FILE TO LATER USE
    # Opening a file
    # file1 = open('network_output.txt', 'a')
    # s = "Batch Size Train," + "Dropout Rate," + "Learning Rate," + "Momentum," + "Train Loss," + "Validation Loss," "Epoch," + "Accuracy" + "\n"

    # # Writing the stats to file
    # file1.write(s)

    # # Closing file
    # file1.close()


    num_of_runs = 0
    total_runs = len(batch_size_train_list)*len(dropout_rate_list)*len(learning_rate_list)\
        *len(momentum_list)*len(n_epoch_list)
    for j in batch_size_train_list:
        batch_size_train = j
        for k in dropout_rate_list:
            dropout_rate = k
            for l in learning_rate_list:
                learning_rate = l
                for m in momentum_list:
                    momentum = m
                    for i in n_epoch_list:
                        n_epochs = i
                        main(sys.argv)
                        num_of_runs = num_of_runs + 1
                        print(" ################################ RUN NO: ", num_of_runs, "/", total_runs, " ################################") 
                        
