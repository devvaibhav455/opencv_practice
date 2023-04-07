"""
Dev Vaibhav
Spring 2023 CS 5330
Project 5: Recognition using Deep Networks 
Task3: Transfer Learning on Greek Letters 
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
from torchsummary import summary
# Helpful to understand imports: https://redirect.cs.umbc.edu/courses/331/fall11/notes/python/python3.ppt.pdf
# Import classes and functions from helper.py
from helper import *
import os
from PIL import Image
import numpy as np
import cv2

# Main function which instantiates an object of the MyNetwork class , loads the weight from the already trained model from tasks 1A to 1E and tries to learn three greek letters alpha, beta and gamma from a total of 27 images. I renamed the model name manually and load model_1A_1E.
def main(argv):
    # print("Arg 1 is: ", argv[1])
    # main function code
    network = MyNetwork()
    network.load_state_dict(torch.load("./results/model_1A_1E.pth"))

   
    # freezes the parameters for the whole network
    # Useful resource: https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6
    for param in network.parameters():
        param.requires_grad = False   

    # Replace the last layer with a new Linear layer with three nodes
    # Modifying the last layer so that it has only three outputs
    network.fc2 = nn.Linear(in_features=50, out_features=3)
    
    #Unfreezing the parameters of the newly added layer. Though, I think this does not need to be done.
    network.fc2.weight.requires_grad = True
    
    # Set the optimizer
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    

    # print("Model after changing is: ", network)

    # Debug trace to print out the requires_grad flag, and layer information
    for param, (name,layer) in zip(network.parameters(), network.named_modules()):
      print(param.requires_grad, name, layer)

    summary(network, (1,28,28))
    #If dataset is already downloaded, it is not downloaded again.
    # DataLoader for the Greek data set


    
    folder_train = "./greek_train/"
    folder_test  = "./greek_test/"
    
    transform = GreekTransform()
    # A set of operations to apply on the handwritten images to pass to the network
    convert_tensor = torchvision.transforms.Compose([
        # torchvision.transforms.ToTensor(), #Converts [0,255] to [0,1]
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(28, antialias=True),
        torchvision.transforms.RandomInvert(p=1.0),
        ThresholdTransform(thr_255=85),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]
    )    

    if len(argv) != 1:
        if argv[1] == "1": #Handwritten alpha, beta, gamma test
            print("Testing for handwritten_greek_test")
            folder_test_no_class  = "./handwritten_greek_test/"
        elif argv[1] == "2": #Extension: Handwritten pi, theta, omega train and test
            print("Testing for handwritten_greek_extra")
            transform = convert_tensor
            # omega is 0 | pi is 1 | theta is 2
            folder_train = "./handwritten_greek_extra_train/"
            folder_test  = "./handwritten_greek_extra_val/"

    

    greek_train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder( folder_train, transform = torchvision.transforms.Compose( [
                torchvision.transforms.ToTensor(),
                transform,
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size = 5, shuffle = True)
    
    greek_test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder( folder_test, transform = torchvision.transforms.Compose( [
                torchvision.transforms.ToTensor(),
                transform,
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size = 1, shuffle = True)

    #https://www.geeksforgeeks.org/how-to-use-a-dataloader-in-pytorch/
    for X, y in greek_train_loader:
        print(f"Shape of X [N, C, H, W]: {X.shape}")
        # print("X min: ", X)
        # print("X max: ", torch.max(X))
        print("Y: ", y)
        print(f"Shape of y: {y.shape} {y.dtype}")
        break

    examples = enumerate(greek_train_loader)
    # print("Batch index: ", list(examples))
    batch_idx, (example_data, example_targets) = next(examples) #Batch no. 0
    # batch_idx, (example_data, example_targets) = next(examples) #Batch no. 1
    # batch_idx, (example_data, example_targets) = next(examples) #Batch no. 2
    # batch_idx, (example_data, example_targets) = next(examples) #Batch no. 3
    # batch_idx, (example_data, example_targets) = next(examples) #Batch no. 4
    # batch_idx, (example_data, example_targets) = next(examples) #Batch no. 5

    #Plotting the letters from the first batch only
    fig = plt.figure()
    for i in range(5):
        plt.subplot(2,3,i+1)
        plt.tight_layout()
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
        plt.title("Ground Truth: {}".format(example_targets[i]))
        plt.xticks([])
        plt.yticks([])
    fig
    plt.show()

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(greek_test_loader.dataset) for i in range(n_epochs + 1)]

    test_network(network, greek_test_loader, test_losses)
    for epoch in range(1, n_epochs + 1):
        train_network(epoch, network, greek_train_loader, optimizer, train_losses, train_counter)
        test_network(network, greek_test_loader, test_losses)
    fig = plt.figure()
    plt.plot(train_counter, train_losses, color='blue')
    plt.scatter(test_counter, test_losses, color='red')
    plt.legend(['Train Loss', 'Test Loss'], loc='upper right')
    plt.xlabel('number of training examples seen')
    plt.ylabel('negative log likelihood loss')
    fig
    plt.show()

    print("Arg 1 is: ", argv[1])
    #################################################################################
    # TESTING ON THE HANDWRITTEN GREEK
    #################################################################################
    if argv[1] == "1":
        network = MyNetwork_Greek()
        network.load_state_dict(torch.load("./results/model_greek.pth"))
        # Model is set in evaluation mode
        network.eval()

        # Initializing the batch with zeros
        batch = torch.zeros(20, 1,28,28)#,dtype=float)
        
        # A set of operations to apply on the handwritten images to pass to the network
        convert_tensor = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(), #Converts [0,255] to [0,1]
            torchvision.transforms.Grayscale(),
            torchvision.transforms.Resize(28, antialias=True),
            torchvision.transforms.RandomInvert(p=1.0),
            ThresholdTransform(thr_255=85),
            torchvision.transforms.Normalize((0.1307,), (0.3081,))
        ]
        )
        
        # Loading all the images inside the folder folder_test
        i = 0
        for filename in os.listdir(folder_test_no_class):
            img = Image.open(os.path.join(folder_test_no_class,filename))
            # cv_img = np.array(img)
            # cv_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
            # cv_img = cv2.threshold(cv_img, 0, 255, cv2.THRESH_OTSU)
            # img = Image.fromarray(cv_img, mode ="") #Convert the cv::Mat to PIL image which is fine
            img = convert_tensor(img) #img is 0/ 1 here after binary
            # print("img is:" , img)
            batch[i] = img
            i = i+1
    
        print("Batch shape: ", batch.shape)

        # Get the NW output on handwritten digits
        with torch.no_grad():
            handwritten_output = network(batch)
        
        # Visualizing the effect of 10 filters of the first layer on the first training example
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
    n_epochs = 5 #72
    batch_size_train = 7
    batch_size_test = 2
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False #cuDNN uses nondeterministic algorithms which can be disabled 
    torch.manual_seed(random_seed)

    main(sys.argv)
