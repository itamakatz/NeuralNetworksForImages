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
import time
import os
import argparse
from contextlib import redirect_stdout

from utils import Running_Time

EPOCHS = 2
# EPOCHS = 35
# EPOCHS = 70
# EPOCHS = 7

DOWNLOAD_FLAG = False
# DOWNLOAD_FLAG = True

DEBUG_PRINT = False
# DEBUG_PRINT = True

DEBUG_FAST_EXECUTION = False
# DEBUG_FAST_EXECUTION = True

SAVE_MODEL_DIR_PATH = r'./SavedModels/'
MODEL_SUFFIX = '.pth'
FIGURE_SUFFIX = '.png'
INFO_SUFFIX = '.txt'

run_time = Running_Time()

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

class SavingPath():

  def __init__(self, args):
    SavingPath._lr = args.lr
    if(args.debug_mode):
      SavingPath._save_name = "_Debugging"
    else:
      if(args.save_name):
        SavingPath._save_name = "_" + args.save_name
      else:
        SavingPath._save_name = ""

    if not os.path.exists(SAVE_MODEL_DIR_PATH):
      os.makedirs(SAVE_MODEL_DIR_PATH)

    SavingPath._path = SAVE_MODEL_DIR_PATH + time.strftime('%b-%d-%Y_%H.%M.%S', time.localtime()) + SavingPath._save_name + "_lr-" + str(SavingPath._lr)
    
  @staticmethod
  def get_path(epoch = -1, suffix = ""):
    if(epoch == -1):
      return SavingPath._path + suffix
    else:
      return SavingPath._path +"_epoch-" + str(epoch) + suffix

class PlotData():
  def __init__(self, length):
    self.index = 0
    self.__maxindex = length
    self.valuesList = np.zeros((2, length))
  def isFull(self):
    return self.index >= self.__maxindex
class ModelStatistics():

  def __init__(self, length, plotNames):
    # plt.figure()
    self.fig, self.ax = plt.subplots()
    self.listDict = {}
    for name in plotNames:
      self.listDict[name] = PlotData(length)

  def AddData(self, name, x, y):
    if(self.listDict[name].isFull()):
      raise Exception("Array is full. Possibly initialized with the wrong length")
    self.listDict[name].valuesList[0][self.listDict[name].index] = x
    self.listDict[name].valuesList[1][self.listDict[name].index] = y
    self.listDict[name].index = self.listDict[name].index + 1

  def Show(self):
    self.ax.cla()
    for name in self.listDict.keys():
      self.ax.plot(self.listDict[name].valuesList[0][:self.listDict[name].index], self.listDict[name].valuesList[1][:self.listDict[name].index], label=name)
    self.fig.legend()
    self.fig.show()

  def Save(self, path):
    self.ax.cla()
    for name in self.listDict.keys():
      self.ax.plot(self.listDict[name].valuesList[0][:self.listDict[name].index], self.listDict[name].valuesList[1][:self.listDict[name].index], label=name)
    self.fig.legend()
    self.fig.savefig(path)

class Net(nn.Module):

  def __init__(self, args):
    super(Net, self).__init__()

    if(torch.cuda.is_available()):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.to(device)

    self.set_model()
    self.lr = args.lr
    self.momentum = args.momentum
    self.loss_function = nn.CrossEntropyLoss()
    if(torch.cuda.is_available()):
      self.loss_function .cuda()
    self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)

  # def Get_loss_function(self):
  #   return self._loss_function

  # def Get_optimizer(self):
  #   return self._optimizer

  def print_shape(self, x, msg, print_flag):
    if(print_flag):
      print(str(x.shape) + " " + msg)

# 1
#   def set_model(self):
#     self.conv1 = nn.Conv2d(3, 6, 5)
#     self.conv2 = nn.Conv2d(6, 16, 5)
#     self.pool = nn.MaxPool2d(2, 2)
#     self.fc1 = nn.Linear(16 * 5 * 5, 120)
#     self.fc2 = nn.Linear(120, 84)
#     self.fc3 = nn.Linear(84, 10)

#   def forward(self, x):
#       x = self.pool(F.relu(self.conv1(x)))
#       x = self.pool(F.relu(self.conv2(x)))
#       x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
#       x = F.relu(self.fc1(x))
#       x = F.relu(self.fc2(x))
#       x = self.fc3(x)
#       return x

# 2
  # def set_model(self):
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

# 3
  # def set_model(self):
  #   self.conv1 = nn.Conv2d(3, 6, 5)
  #   self.conv2 = nn.Conv2d(6, 16, 5)
  #   self.pool = nn.MaxPool2d(2, 2)
  #   self.fc1 = nn.Linear(16 * 5 * 5, 84)
  #   # self.fc2 = nn.Linear(120, 84)
  #   self.fc3 = nn.Linear(84, 10)

  # def forward(self, x):
  #     x = self.pool(F.relu(self.conv1(x)))
  #     x = self.pool(F.relu(self.conv2(x)))
  #     x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
  #     x = F.relu(self.fc1(x))
  #     # x = F.relu(self.fc2(x))
  #     x = self.fc3(x)
  #     return x

# 4
  def set_model(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 10)
    # self.fc3 = nn.Linear(84, 10)

  def forward(self, x):
      x = self.pool(F.relu(self.conv1(x)))
      x = self.pool(F.relu(self.conv2(x)))
      x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
      x = F.relu(self.fc1(x))
      # x = F.relu(self.fc2(x))
      x = self.fc2(x)
      return x

def train(net: Net, trainloader, epoch):
  net.train()
  running_loss = 0.0
  epoch_loss = 0.0
  for i, data in enumerate(trainloader, 0):
    # get the inputs; data is a list of [inputs, labels]
    inputs, labels = data

    if(torch.cuda.is_available()):
      inputs, labels = inputs.cuda(), labels.cuda()

    # zero the parameter gradients
    net.optimizer.zero_grad()

    # forward + backward + optimize
    outputs = net(inputs)
    loss = net.loss_function(outputs, labels)
    loss.backward()
    net.optimizer.step()

    # print statistics
    running_loss += loss.item()
    epoch_loss += loss.item()
    if i % 2000 == 1999: # print every 2000 mini-batches
      print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0
      if(DEBUG_FAST_EXECUTION): 
        break

  return epoch_loss/ len(trainloader)

def test(net: Net, testloader):
  net.eval()
  correct = 0
  total = 0

  with torch.no_grad():
    for data in testloader:

      images, labels = data
      if(torch.cuda.is_available()):
        images, labels = images.cuda(), labels.cuda()

      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  return correct/total

def run(net: Net, trainloader, testloader):

  modelStatistics = ModelStatistics(EPOCHS, ["Train Loss", "Test Accuracy"])
  modelStatistics.ax.set_title("Training")
  modelStatistics.ax.set_xlabel("Epochs")
  # modelStatistics.ax.set_ylabel("Training Loss")

  for epoch in range(EPOCHS):  # loop over the dataset multiple times

    epoch_loss = train(net, trainloader, epoch)
    test_accuracy = test(net, testloader)

    modelStatistics.AddData("Train Loss", epoch, epoch_loss )
    modelStatistics.AddData("Test Accuracy", epoch, test_accuracy)

    # modelStatistics.Show()

    modelStatistics.Save(SavingPath.get_path(epoch, FIGURE_SUFFIX))
    torch.save(net.state_dict(), SavingPath.get_path(epoch, MODEL_SUFFIX))
    # modelStatistics.Save("_epochs-" + str(epoch) + FIGURE_SUFFIX)
    # torch.save(net.state_dict(), PATH + "_lr-" + str(net.lr) + "_epochs-" + str(epoch) + MODEL_SUFFIX) 
    # save trained model. See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_ for more details on saving PyTorch models.
    print(f"Running time: {run_time.get_running_time()}")

  print('Finished Training')


def final_test(testloader, classes):
  net = Net()
  if(torch.cuda.is_available()):
    net.cuda() 

  net.load_state_dict(torch.load(PATH)) # todo ================================================================================ fix

  correct = 0
  total = 0

  testStatistics = ModelStatistics(len(testloader), ["test"])
  testStatistics.ax.set_title("Testing")
  testStatistics.ax.set_xlabel("Test Data")
  testStatistics.ax.set_ylabel("Accuracy")

  with torch.no_grad():
    for data in testloader:

      images, labels = data
      if(torch.cuda.is_available()):
        images, labels = images.cuda(), labels.cuda()

      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()
      testStatistics.AddData("test", total, correct/total)

  testStatistics.Show()
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
  
def main(args):

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

  net = Net(args)
  if(torch.cuda.is_available()):
    net.cuda() 

  # batch_size = len(trainloader)
  # # get some random training images
  # dataiter = iter(trainloader)
  # images, labels = dataiter.next()
  # bla = len(images)
  # im = images[0]
  # summary(net)

  with open(SavingPath.get_path(suffix=INFO_SUFFIX) , 'w') as f:
    with redirect_stdout(f):
      print(net)
      summary(net, input_size=(3, 32, 32))
      print('\nExtra info:')
      print("Model's state_dict:")
      for param_tensor in net.state_dict():
        print(param_tensor, "\t", net.state_dict()[param_tensor].size())
      print()
      # Print optimizer's state_dict
      print("Optimizer's state_dict:")
      for var_name in net.optimizer.state_dict():
        print(var_name, "\t", net.optimizer.state_dict()[var_name])

  # summary(net, (3, 32, 32), depth=3)

  run(net, trainloader, testloader)
  # test(testloader, classes)

if __name__ == '__main__':

  run_time = Running_Time()

  parser = argparse.ArgumentParser(description='Test cifar10')
  parser.add_argument('--epochs', type=int, default=4, metavar='N',
                      help='number of epochs to train (default: 4)')
  parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                      help='learning rate (default: 0.001)')
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                      help='SGD momentum (default: 0.9)')
  parser.add_argument('--save-name', default="",
                      help='Specific name when saving the model')
  parser.add_argument('--debug-mode', action='store_true', default=False,
                      help='Switches to debug mode which overwrites all other given arguments (default: False)')
  args = parser.parse_args()

  # PATH = SAVE_MODEL_DIR_PATH + 'net_' + time.strftime('%b-%d-%Y_%H.%M.%S', time.localtime())
  if(args.debug_mode):
    print('\n*************************************************')
    print('************* Running in DEBUG mode *************')
    print('*************************************************\n')
    EPOCHS = 4
    # PATH = PATH + '_Debuging'
    DEBUG_FAST_EXECUTION = True
  else:
    EPOCHS = args.epochs
    # if(args.save_name):
      # PATH = PATH + '_' + args.save_name
    DEBUG_FAST_EXECUTION = False

  SavingPath(args)
  main(args)

  print("Total Execution Time:")
  print(f"--- {run_time.get_running_time()} ---")

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
