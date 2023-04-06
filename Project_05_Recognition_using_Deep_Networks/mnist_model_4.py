"""
Dev Vaibhav
Spring 2023 CS 5330
Project 5: Recognition using Deep Networks 
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


# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        
        super(MyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d(dropout_rate)
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=10)

        
        

    # computes a forward pass for the network
    # methods need a summary comment
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)),2))
        x = x.view(-1,320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training) #During training, randomly zeroes some of the elements of the input tensor with probability p using samples from a Bernoulli distribution. Default probability: 0.5
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# useful functions with a comment for each function
# Src: https://medium.com/@nutanbhogendrasharma/pytorch-convolutional-neural-network-with-mnist-dataset-4e8a4265e118
def train_network( epoch, network, train_loader, optimizer, train_losses, train_counter ):
    
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), './results/model.pth')
            torch.save(optimizer.state_dict(), './results/optimizer.pth')
    return

def test_network(network, test_loader, test_losses):
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      output = network(data)
      test_loss += F.nll_loss(output, target, size_average=False).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))
  
  #return the accuracy
  return(100. * correct / len(test_loader.dataset))


# main function (yes, it needs a comment too)
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

    train_losses = []
    train_counter = []
    test_losses = []
    test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]
    accuracies = [] #Accuracy for each epoch after network is tested on test set

    accuracies.append(test_network(network, test_loader, test_losses))
    
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
                        
