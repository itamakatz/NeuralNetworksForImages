import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchsummary import summary
import argparse
from contextlib import redirect_stdout

from utils import Running_Time, ModelStatistics, SavingPath, PlotData
from models import Net

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ magic numbers/strings ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

EPOCHS = 10
EPOCHS_TO_SAVE = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]

DOWNLOAD_FLAG = False
# DOWNLOAD_FLAG = True

DEBUG_FAST_EXECUTION = False
# DEBUG_FAST_EXECUTION = True

SAVE_DIR_NAME = 'SavedModels'
MODEL_SUFFIX = '.pth'
FIGURE_SUFFIX = '.png'
INFO_SUFFIX = '.txt'

run_time = Running_Time()

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

# training
def train(net: Net, trainloader, epoch):
  ''' trains the net given a set of inputs and outputs '''

  net.train()
  running_loss = 0.0
  train_loss = 0.0
  correct = 0
  total = 0
  for i, data in enumerate(trainloader, 0):
    x, y = data

    if(torch.cuda.is_available()):
      x, y = x.cuda(), y.cuda()

    # zero the parameter gradients
    net.optimizer.zero_grad()

    outputs = net(x)
    loss = net.loss_function(outputs, y)
    loss.backward()
    net.optimizer.step()

    running_loss += loss.item()
    train_loss += loss.item()
    _, predicted = torch.max(outputs.data, 1)
    total += y.size(0)
    correct += (predicted == y).sum().item()

    # print sub-epoch statistics
    if i % 2000 == 1999: # print every 2000 mini-batches
      print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
      running_loss = 0.0
      if(DEBUG_FAST_EXECUTION): 
        break

  return (train_loss/ len(trainloader),  correct/total)

# 
def test(net: Net, testloader):
  ''' validates the net given a set of inputs and outputs '''

  net.eval()
  correct = 0
  total = 0
  test_loss = 0
  with torch.no_grad():
    for data in testloader:

      x, y = data
      if(torch.cuda.is_available()): x, y = x.cuda(), y.cuda()
      
      outputs = net(x)
      loss = net.loss_function(outputs, y)
      test_loss += loss.item()
      _, predicted = torch.max(outputs.data, 1)
      total += y.size(0)
      correct += (predicted == y).sum().item()
    test_loss /= len(testloader)
  return (test_loss, correct/total)

def run(net: Net, trainloader, testloader):
  ''' runs the test and validation for each ephoch as well as saves figures, 
      the model and updates a statistics text file '''

  modelStatistics = ModelStatistics(EPOCHS, ["Train Loss", "Train Accuracy", "Test Loss", "Test Accuracy"])

  for epoch in range(EPOCHS):  # loop over the dataset multiple times

    train_loss , train_accuracy = train(net, trainloader, epoch)
    test_loss, test_accuracy = test(net, testloader)

    modelStatistics.addData("Train Loss", epoch, train_loss)
    modelStatistics.addData("Train Accuracy", epoch, train_accuracy)
    modelStatistics.addData("Test Loss", epoch, test_loss)
    modelStatistics.addData("Test Accuracy", epoch, test_accuracy)

    # save the current statistics to a file
    with open(SavingPath.get_path(suffix=INFO_SUFFIX) , 'a+') as f:
      with redirect_stdout(f):
        print(f'epoch: {epoch+1}, train_loss: {train_loss}, train_accuracy: {train_accuracy}, test_loss: {test_loss}, test_accuracy: {test_accuracy}')

    if((epoch + 1) in EPOCHS_TO_SAVE):
      modelStatistics.save(SavingPath.get_path(epoch+1, FIGURE_SUFFIX), f"{str(net.model_name)}. Epoch-{(epoch + 1)}")
      torch.save(net.state_dict(), SavingPath.get_path(epoch+1, MODEL_SUFFIX))

    # save trained model. See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_ for more details on saving PyTorch models.
    print(f'epoch: {epoch+1}, train_loss: {train_loss}, train_accuracy: {train_accuracy}, test_loss: {test_loss}, test_accuracy: {test_accuracy}. Running time: {run_time.get_running_time()}')

  print('Finished Training')

def final_test(net, testloader, classes):
  ''' final test that prints out also the statistics over each 
      of the 10 classes '''

  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))
  with torch.no_grad():
    for data in testloader:
      images, labels = data
      if(torch.cuda.is_available()):
        images, labels = images.cuda(), labels.cuda()
      outputs = net(images)
      _, predicted = torch.max(outputs, 1)
      c = (predicted == labels).squeeze()
      for i in range(4):
        label = labels[i]
        class_correct[label] += c[i].item()
        class_total[label] += 1

  with open(SavingPath.get_path(suffix=INFO_SUFFIX) , 'a+') as f:
    with redirect_stdout(f):
      print()
      for i in range(10):
        print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
  
def main(args):
  ''' initializes everything before starting lo train the net '''

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

  # save some info about the model into a text file
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
      print()

  run(net, trainloader, testloader)
  final_test(net, testloader, classes)

if __name__ == '__main__':

  run_time = Running_Time()

  # parse the user input

  parser = argparse.ArgumentParser(description='Test cifar10')
  parser.add_argument('--epochs', type=int, default=10, metavar='e',
                      help='number of epochs to train (default: 10)')
  parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                      help='learning rate (default: 0.001)')
  parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                      help='SGD momentum (default: 0.9)')
  parser.add_argument('--model-name', default="original", metavar='N',
                      help='Chooses the model type. (default: "original")')
  parser.add_argument('--debug-mode', action='store_true', default=False,
                      help='Switches to debug mode which overwrites all other given arguments (default: False)')
  args = parser.parse_args()

  if(args.debug_mode):
    print('\n*************************************************')
    print('************* Running in DEBUG mode *************')
    print('*************************************************\n')
    EPOCHS = 5
    DEBUG_FAST_EXECUTION = True
  else:
    EPOCHS = args.epochs
    DEBUG_FAST_EXECUTION = False

  SavingPath(args, SAVE_DIR_NAME)

  main(args) # start the training!!

  print("Total Execution Time:")
  print(f"--- {run_time.get_running_time()} ---")