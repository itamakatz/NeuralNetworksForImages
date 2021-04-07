import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import torch.optim as optim

EPOCHS = 10
# EPOCHS = 70
# EPOCHS = 7

DOWNLOAD_FLAG = False
# DOWNLOAD_FLAG = True

DEBUG_PRINT = False
# DEBUG_PRINT = True

PATH = './cifar_net.pth'


# functions to show an image
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# if __name__ == '__main__':
#   # get some random training images
#   data_iter = iter(trainloader)
#   images, labels = data_iter.next()

#   # show images
#   imshow(torchvision.utils.make_grid(images))
#   # print labels
#   print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

class PlotAccuracy():
  
  def __init__(self, length):
    # plt.figure()
    self.fig, self.ax = plt.subplots()
    self.index = 0
    self.__maxindex = length
    self.valuesList = np.zeros((2, length))

  def AddData(self, x, y):
    if(self.index >= self.__maxindex):
      raise Exception("Array is full. Possibly initialized with the wrong length")
    self.valuesList[0][self.index] = x
    self.valuesList[1][self.index] = y
    self.index = self.index + 1
    if(self.index == self.__maxindex):
      self.ax.plot(self.valuesList[0], self.valuesList[1])

  # def Show(self, label="", xlabel="", ylabel=""):
  def Show(self):
    # plt.clf() # clears the entire current figure
    # self.ax.plot(self.valuesList[0], self.valuesList[1], label=label)
    # if(not xlabel):
    #   self.ax.set_xlabel(xlabel)
    # if(not ylabel):
    #   self.ax.set_ylabel(ylabel)
    self.fig.show()

  def Save(self, path):
  # def Save(self, path, label="", xlabel="", ylabel=""):
    # plt.clf() # clears the entire current figure
    # self.ax.plot(self.valuesList[0], self.valuesList[1], label=label)
    # if(not xlabel):
    #   self.ax.set_xlabel(xlabel)
    # if(not ylabel):
    #   self.ax.set_ylabel(ylabel)
    self.fig.savefig(path)

class Net(nn.Module):

  def __init__(self):
    super(Net, self).__init__()

    if(torch.cuda.is_available()):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.to(device)

    self.set_model()

  def Get_loss_function(self):
    if(torch.cuda.is_available()):
      return nn.CrossEntropyLoss().cuda()
    else:
      return nn.CrossEntropyLoss()

  def Get_optimizer(self):
    return optim.SGD(self.parameters(), lr=0.001, momentum=0.9)

  def print_shape(self, x, msg, print_flag):
    if(print_flag):
      print(str(x.shape) + " " + msg)


# 1.
  def set_model(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return x

# 2
  # def print_shape(self):
  #     self.conv1 = nn.Conv2d(3, 6, 3)
  #     self.conv2 = nn.Conv2d(6, 6, 3, padding=1)
  #     self.conv3 = nn.Conv2d(6, 16, 5)
  #     self.pool = nn.MaxPool2d(2, 2)
  #     self.fc1 = nn.Linear(16 * 5 * 5, 120)
  #     self.fc2 = nn.Linear(120, 84)
  #     self.fc3 = nn.Linear(84, 10)

  # def forward(self, x):
  #     self.print_shape(x, "input", DEBUG_PRINT)
  #     x = F.relu(self.conv1(x))
  #     self.print_shape(x, "conv1", DEBUG_PRINT)
  #     x = self.pool(x)
  #     self.print_shape(x, "pool", DEBUG_PRINT)
  #     for i in range(5):
  #       x = F.relu(self.conv2(x))
  #       self.print_shape(x, "conv2", DEBUG_PRINT)
  #     x = F.relu(self.conv3(x))
  #     self.print_shape(x, "conv3", DEBUG_PRINT)
  #     x = self.pool(x)
  #     self.print_shape(x, "pool", DEBUG_PRINT)
  #     x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
  #     self.print_shape(x, "view", DEBUG_PRINT)
  #     x = F.relu(self.fc1(x))
  #     self.print_shape(x, "fc1", DEBUG_PRINT)
  #     x = F.relu(self.fc2(x))
  #     self.print_shape(x, "fc2", DEBUG_PRINT)
  #     x = self.fc3(x)
  #     self.print_shape(x, "fc3", DEBUG_PRINT)
  #     return x

# 3.
  # def print_shape(self):
  #     self.conv1 = nn.Conv2d(3, 6, 3)
  #     self.conv2 = nn.Conv2d(6, 12, 3)
  #     self.conv3 = nn.Conv2d(12, 20, 3)
  #     self.pool = nn.MaxPool2d(2, 2)
  #     self.fc1 = nn.Linear(20 * 3 * 3, 100)
  #     self.fc2 = nn.Linear(100, 64)
  #     self.fc3 = nn.Linear(64, 10)

  # def forward(self, x):
  #     self.print_shape(x, "input", DEBUG_PRINT)
  #     x = F.relu(self.conv1(x))
  #     self.print_shape(x, "conv1", DEBUG_PRINT)
  #     x = self.pool(x)
  #     self.print_shape(x, "pool", DEBUG_PRINT)
  #     x = F.relu(self.conv2(x))
  #     self.print_shape(x, "conv2", DEBUG_PRINT)
  #     x = self.pool(x)
  #     self.print_shape(x, "pool", DEBUG_PRINT)
  #     x = F.relu(self.conv3(x))
  #     self.print_shape(x, "conv3", DEBUG_PRINT)
  #     x = self.pool(x)
  #     self.print_shape(x, "pool", DEBUG_PRINT)
  #     # x = x.view(-1, 20 * 3 * 3) # reshapes for the fully connected
  #     view_size = 1
  #     for i in range(len(x.shape)):
  #       view_size = view_size*x.shape[i]
  #     x = x.view(view_size) # reshapes for the fully connected
  #     self.print_shape(x, "view", DEBUG_PRINT)
  #     x = F.relu(self.fc1(x))
  #     self.print_shape(x, "fc1", DEBUG_PRINT)
  #     x = F.relu(self.fc2(x))
  #     self.print_shape(x, "fc2", DEBUG_PRINT)
  #     x = self.fc3(x)
  #     self.print_shape(x, "fc3", DEBUG_PRINT)
  #     return x

def train(net: Net, trainloader):

  criterion = net.Get_loss_function()
  optimizer = net.Get_optimizer()
  plotAccuracy = PlotAccuracy(EPOCHS)
  plotAccuracy.ax.set_title("Training")
  plotAccuracy.ax.set_xlabel("Epochs")
  plotAccuracy.ax.set_ylabel("Training Loss")

  for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    epoch_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      inputs, labels = data

      if(torch.cuda.is_available()):
        inputs, labels = inputs.cuda(), labels.cuda()

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      epoch_loss += loss.item()
      if i % 2000 == 1999: # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

    plotAccuracy.AddData(epoch, epoch_loss / len(trainloader))
    epoch_loss = 0.0

  plotAccuracy.Show()

  print('Finished Training')

  # save trained model. See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_ for more details on saving PyTorch models.
  torch.save(net.state_dict(), PATH)

def test(testloader, classes):

  dataiter = iter(testloader)
  images, labels = dataiter.next()

  net = Net()
  # if(torch.cuda.is_available()):
  #   net.cuda() 

  net.load_state_dict(torch.load(PATH))

  outputs = net(images)

  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] for j in range(4)))

  correct = 0
  total = 0
  plotAccuracy = PlotAccuracy(len(testloader))
  plotAccuracy.ax.set_title("Testing")
  plotAccuracy.ax.set_xlabel("Test Data")
  plotAccuracy.ax.set_ylabel("Accuracy")

  with torch.no_grad():
    for data in testloader:

      images, labels = data

      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      plotAccuracy.AddData(total, correct/total)

  plotAccuracy.Show()
  print('Accuracy of the network on the %d test images: %d %%' % (len(testloader) * testloader.batch_size, 100 * correct / total))

  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1


  for i in range(10):
    print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
  
  del dataiter

def main():

  # ◄►◄► Set CUDA GPU ◄►◄► #
  if(torch.cuda.is_available()):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      torch.cuda.set_device(device)
      print("is CUDA initialized: "+ "YES" if torch.cuda.is_initialized() else "NO")

  # ◄►◄► Download data ◄►◄► #
  transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=DOWNLOAD_FLAG, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=DOWNLOAD_FLAG, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

  net = Net()
  if(torch.cuda.is_available()):
    net.cuda() 

  # batch_size = len(trainloader)
  # # get some random training images
  # dataiter = iter(trainloader)
  # images, labels = dataiter.next()
  # bla = len(images)
  # im = images[0]
  # summary(net)
  summary(net, input_size=(3, 32, 32))
  # summary(net, (3, 32, 32), depth=3)

  train(net, trainloader)
  test(testloader, classes)

if __name__ == '__main__':
  main()
  _ = input("Press enter to finish..")


'''
Documentation:

  1. For the normal net

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 28, 28]             456
             MaxPool2d-2            [-1, 6, 14, 14]               0
                Conv2d-3           [-1, 16, 10, 10]           2,416
             MaxPool2d-4             [-1, 16, 5, 5]               0
                Linear-5                  [-1, 120]          48,120
                Linear-6                   [-1, 84]          10,164
                Linear-7                   [-1, 10]             850
    ================================================================
    Total params: 62,006
    Trainable params: 62,006
    Non-trainable params: 0

    Accuracy of the network on the 10000 test images: 50 %
    Accuracy of plane : 67 %
    Accuracy of   car : 36 %
    Accuracy of  bird : 32 %
    Accuracy of   cat : 37 %
    Accuracy of  deer : 40 %
    Accuracy of   dog : 49 %
    Accuracy of  frog : 28 %
    Accuracy of horse : 73 %
    Accuracy of truck : 66 %

  2. With 7 epochs

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 28, 28]             456
             MaxPool2d-2            [-1, 6, 14, 14]               0
                Conv2d-3           [-1, 16, 10, 10]           2,416
             MaxPool2d-4             [-1, 16, 5, 5]               0
                Linear-5                  [-1, 120]          48,120
                Linear-6                   [-1, 84]          10,164
                Linear-7                   [-1, 10]             850
    ================================================================
    Total params: 62,006
    Trainable params: 62,006
    Non-trainable params: 0

    Accuracy of the network on the 10000 test images: 61 %
    Accuracy of plane : 62 %
    Accuracy of   car : 84 %
    Accuracy of  bird : 52 %
    Accuracy of   cat : 37 %
    Accuracy of  deer : 52 %
    Accuracy of   dog : 58 %
    Accuracy of  frog : 73 %
    Accuracy of horse : 56 %
    Accuracy of  ship : 68 %
    Accuracy of truck : 64 %

  3. Many small conv2:

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1            [-1, 6, 30, 30]             168
            MaxPool2d-2            [-1, 6, 15, 15]               0
                Conv2d-3            [-1, 6, 15, 15]             330
                Conv2d-4            [-1, 6, 15, 15]             330
                Conv2d-5            [-1, 6, 15, 15]             330
                Conv2d-6            [-1, 6, 15, 15]             330
                Conv2d-7            [-1, 6, 15, 15]             330
                Conv2d-8           [-1, 16, 11, 11]           2,416
            MaxPool2d-9             [-1, 16, 5, 5]               0
              Linear-10                  [-1, 120]          48,120
              Linear-11                   [-1, 84]          10,164
              Linear-12                   [-1, 10]             850
    ================================================================
    Total params: 63,368
    Trainable params: 63,368
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 0.12
    Params size (MB): 0.24
    Estimated Total Size (MB): 0.38
    ----------------------------------------------------------------
    [1,  2000] loss: 2.304
    [1,  4000] loss: 2.303
    [1,  6000] loss: 2.303
    [1,  8000] loss: 2.304
    [1, 10000] loss: 2.303
    [1, 12000] loss: 2.304
    [2,  2000] loss: 2.303
    [2,  4000] loss: 2.303
    [2,  6000] loss: 2.304
    [2,  8000] loss: 2.303
    [2, 10000] loss: 2.303
    [2, 12000] loss: 2.304
    [3,  2000] loss: 2.303
    [3,  4000] loss: 2.303
    [3,  6000] loss: 2.303
    [3,  8000] loss: 2.303
    [3, 10000] loss: 2.303
    [3, 12000] loss: 2.303
    [4,  2000] loss: 2.303
    [4,  4000] loss: 2.303
    [4,  6000] loss: 2.303
    [4,  8000] loss: 2.302
    [4, 10000] loss: 2.264
    [4, 12000] loss: 2.016
    [5,  2000] loss: 1.825
    [5,  4000] loss: 1.759
    [5,  6000] loss: 1.705
    [5,  8000] loss: 1.635
    [5, 10000] loss: 1.605
    [5, 12000] loss: 1.576
    [6,  2000] loss: 1.558
    [6,  4000] loss: 1.539
    [6,  6000] loss: 1.532
    [6,  8000] loss: 1.513
    [6, 10000] loss: 1.487
    [6, 12000] loss: 1.468
    [7,  2000] loss: 1.437
    [7,  4000] loss: 1.463
    [7,  6000] loss: 1.445
    [7,  8000] loss: 1.425
    [7, 10000] loss: 1.425
    [7, 12000] loss: 1.438
    Finished Training
    Predicted:    cat  ship plane plane
    Accuracy of the network on the 10000 test images: 47 %
    Accuracy of plane : 58 %
    Accuracy of   car : 51 %
    Accuracy of  bird : 34 %
    Accuracy of   cat : 19 %
    Accuracy of  deer : 49 %
    Accuracy of   dog : 35 %
    Accuracy of  frog : 62 %
    Accuracy of horse : 57 %
    Accuracy of  ship : 57 %
    Accuracy of truck : 45 %

'''
