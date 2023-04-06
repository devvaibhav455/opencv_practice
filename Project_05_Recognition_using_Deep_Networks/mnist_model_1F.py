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
import os
from PIL import Image

#Src: https://stackoverflow.com/questions/65979207/applying-a-simple-transformation-to-get-a-binary-image-using-pytorch
class ThresholdTransform(object):
  def __init__(self, thr_255):
    self.thr = thr_255 / 255.  # input threshold for [0..255] gray level, convert to [0..1]

  # x is the input image
  def __call__(self, x):
    
    # print("Thr is:" , self.thr)

    return (x > self.thr).to(x.dtype)  # do not change the data type


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


# main function (yes, it needs a comment too)
def main(argv):
    # handle any command line arguments in argv

    # main function code
    network = MyNetwork()
    network.load_state_dict(torch.load("./results/model.pth"))
    network.eval()

    #If dataset is already downloaded, it is not downloaded again.
    # test_loader = torch.utils.data.DataLoader(
    #     torchvision.datasets.MNIST('./files/', train=False, download=True,
    #                                 transform=torchvision.transforms.Compose([
    #                                 torchvision.transforms.ToTensor(),
    #                                 torchvision.transforms.Normalize(
    #                                     (0.1307,), (0.3081,))
    #                                 ])),
    #     batch_size=batch_size_test, shuffle=True)

    # for X, y in test_loader:
    #     print(f"Shape of X [N, C, H, W]: {X.shape}")
    #     print(f"Shape of y: {y.shape} {y.dtype}")
    #     break

    # examples = enumerate(test_loader)
    # batch_idx, (example_data, example_targets) = next(examples)

    # with torch.no_grad():
    #     output = network(example_data)

    # fig = plt.figure()
    # for i in range(6):
    #     plt.subplot(2,3,i+1)
    #     plt.tight_layout()
    #     plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    #     plt.title("Prediction: {}".format(
    #         output.data.max(1, keepdim=True)[1][i].item()))
    #     plt.xticks([])
    #     plt.yticks([])
    # fig
    # plt.show()

    ############################################################################
    # READ USERS HANDWRITTEN IMAGES FROM A FOLDER
    ############################################################################

    #Src: https://stackoverflow.com/questions/30230592/loading-all-images-using-imread-from-a-given-folder;
    # https://www.projectpro.io/recipes/convert-image-tensor-pytorch; 
    # https://www.analyticsvidhya.com/blog/2021/04/10-pytorch-transformations-you-need-to-know/
    my_folder = "./handwritten_digits/"
    convert_tensor = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(), #Converts [0,255] to [0,1]
        torchvision.transforms.Grayscale(),
        torchvision.transforms.Resize(28, antialias=True),
        torchvision.transforms.RandomInvert(p=1.0),
        ThresholdTransform(thr_255=180),
        # torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ]
    )
    batch = torch.zeros(10, 1,28,28)#,dtype=float)

    i = 0
    for filename in os.listdir(my_folder):
        img = Image.open(os.path.join(my_folder,filename))
        img = convert_tensor(img) #img is 0/ 1 here after binary
        # print("img is:" , img)
        batch[i] = img
        i = i+1
    
    print("Batch shape: ", batch.shape)
    

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
    n_epochs = 3
    batch_size_train = 64
    batch_size_test = 1000
    learning_rate = 0.01
    momentum = 0.5
    log_interval = 10

    random_seed = 1
    torch.backends.cudnn.enabled = False #cuDNN uses nondeterministic algorithms which can be disabled 
    torch.manual_seed(random_seed)

    main(sys.argv)
