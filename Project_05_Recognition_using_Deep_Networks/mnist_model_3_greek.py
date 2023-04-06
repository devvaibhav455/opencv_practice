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
from torchsummary import summary


# greek data set transform
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )

# class definitions
class MyNetwork(nn.Module):
    def __init__(self):
        
        super(MyNetwork, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
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
            torch.save(network.state_dict(), './results/model_greek.pth')
            torch.save(optimizer.state_dict(), './results/optimizer_greek.pth')
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

# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    # main function code
    network = MyNetwork()
    network.load_state_dict(torch.load("./results/model.pth"))

    
   
    # Useful resource: https://jimmy-shen.medium.com/pytorch-freeze-part-of-the-layers-4554105e03a6
    # freezes the parameters for the whole network
    for param in network.parameters():
        param.requires_grad = False   

    # network.fc2 = nn.Identity()
    network.fc2 = nn.Linear(in_features=50, out_features=3)
    network.fc2.weight.requires_grad = True
    #Modifying the last layer so that it has only three outputs
    # network = nn.Sequential(network, nn.Linear(in_features=50, out_features=3))
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)
    # optimizer.load_state_dict(torch.load("./results/optimizer.pth"))

    # print("Model after changing is: ", network)

    # Debug trace to print out the requires_grad flag, and layer information
    for param, (name,layer) in zip(network.parameters(), network.named_modules()):
      print(param.requires_grad, name, layer)

    summary(network, (1,28,28))
    #If dataset is already downloaded, it is not downloaded again.
    # DataLoader for the Greek data set
    greek_train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder( "./greek_train/", transform = torchvision.transforms.Compose( [
                torchvision.transforms.ToTensor(),
                GreekTransform(),
                torchvision.transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size = 5, shuffle = True)
    
    greek_test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.ImageFolder( "./greek_test/", transform = torchvision.transforms.Compose( [
                torchvision.transforms.ToTensor(),
                GreekTransform(),
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
    return

if __name__ == "__main__":
    n_epochs = 65
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False #cuDNN uses nondeterministic algorithms which can be disabled 
    torch.manual_seed(random_seed)

    main(sys.argv)
