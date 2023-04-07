from torch import nn
import torch.nn.functional as F
import torch
import torchvision

log_interval = 1 # Capture logs after every 10 batches


#Src: https://stackoverflow.com/questions/65979207/applying-a-simple-transformation-to-get-a-binary-image-using-pytorch
# Binary image conversion based on a threshold value
class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  # x is the input image
  def __call__(self, x):
    # print("Thr is:" , self.thr)
    return (x > self.thr).to(x.dtype)  # do not change the data type
  
# class definitions
# greek data set transform provided by the professor
class GreekTransform:
    def __init__(self):
        pass

    def __call__(self, x):
        x = torchvision.transforms.functional.rgb_to_grayscale( x )
        x = torchvision.transforms.functional.affine( x, 0, (0,0), 36/128, 0 )
        x = torchvision.transforms.functional.center_crop( x, (28, 28) )
        return torchvision.transforms.functional.invert( x )
    
#Src: https://nextjournal.com/gkoehler/pytorch-mnist
# Class defining the MNIST architecture present on the tutorial
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

#Class to detect the greek letters  (Fully connected 2 out features is 3)
class MyNetwork_Greek(nn.Module):
    def __init__(self):
        
        super(MyNetwork_Greek, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(in_channels=10, out_channels=20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(in_features=320, out_features=50)
        self.fc2 = nn.Linear(in_features=50, out_features=3)

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

# Function used to train the network
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


# Function used to test the network
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