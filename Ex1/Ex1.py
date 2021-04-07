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

class Net(nn.Module):

    def print_shape(self, x, msg, print_flag):
      if(print_flag):
        print(str(x.shape) + " " + msg)

# 1.
    def __init__(self):
        super(Net, self).__init__()
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
    # def __init__(self):
    #     super(Net, self).__init__()
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
    # def __init__(self):
    #     super(Net, self).__init__()
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

def main():

  if(torch.cuda.is_available()):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      torch.cuda.set_device(device)
      print("is CUDA initialized: "+ "YES" if torch.cuda.is_initialized() else "NO")

  # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #
  # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► Download Data ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #
  # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

  # EPOCHS = 2
  EPOCHS = 7

  DOWNLOAD_FLAG = False
  # DOWNLOAD_FLAG = True

  # DEBUG_PRINT = False
  DEBUG_PRINT = True

  transform = transforms.Compose(
      [transforms.ToTensor(),
      transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

  trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=DOWNLOAD_FLAG, transform=transform)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

  testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=DOWNLOAD_FLAG, transform=transform)
  testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)

  classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

  # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

  net = Net()

  if(torch.cuda.is_available()):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)

  batch_size = len(trainloader)
  # get some random training images
  dataiter = iter(trainloader)
  images, labels = dataiter.next()
  bla = len(images)
  im = images[0]
  # summary(net)
  summary(net, input_size=(3, 32, 32))
  # summary(net, (3, 32, 32), depth=3)

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

########################################################################
# 4. Train the network
# ^^^^^^^^^^^^^^^^^^^^
#
# This is when things start to get interesting.
# We simply have to loop over our data iterator, and feed the inputs to the
# network and optimize.

  for epoch in range(EPOCHS):  # loop over the dataset multiple times

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):
      # get the inputs; data is a list of [inputs, labels]
      if(torch.cuda.is_available()):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        inputs, labels = data[0].to(device), data[1].to(device)
      else:
        inputs, labels = data

      # zero the parameter gradients
      optimizer.zero_grad()

      # forward + backward + optimize
      outputs = net(inputs)
      loss = criterion(outputs, labels)
      loss.backward()
      optimizer.step()

      # print statistics
      running_loss += loss.item()
      if i % 2000 == 1999:    # print every 2000 mini-batches
        print('[%d, %5d] loss: %.3f' %
          (epoch + 1, i + 1, running_loss / 2000))
        running_loss = 0.0

  print('Finished Training')

  # save trained model:
# See `here <https://pytorch.org/docs/stable/notes/serialization.html>`_ for more details on saving PyTorch models.
  PATH = './cifar_net.pth'
  torch.save(net.state_dict(), PATH)


# 5. Test the network on the test data
#
# We have trained the network for 2 passes over the training dataset.
# But we need to check if the network has learnt anything at all.
#
# We will check this by predicting the class label that the neural network
# outputs, and checking it against the ground-truth. If the prediction is
# correct, we add the sample to the list of correct predictions.
#
# Okay, first step. Let us display an image from the test set to get familiar.

  dataiter = iter(testloader)
  images, labels = dataiter.next()

  # print images
  # imshow(torchvision.utils.make_grid(images))
  # print('GroundTruth: ', ' '.join('%5s' % classes[labels[j]] for j in range(4)))

  ########################################################################
  # Next, let's load back in our saved model (note: saving and re-loading the model
  # wasn't necessary here, we only did it to illustrate how to do so):

  net = Net()
  # if(torch.cuda.is_available()):
  #   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  #   net.to(device)
  net.load_state_dict(torch.load(PATH))

  ########################################################################
  # Okay, now let us see what the neural network thinks these examples above are:

  outputs = net(images)

  ########################################################################
  # The outputs are energies for the 10 classes.
  # The higher the energy for a class, the more the network
  # thinks that the image is of the particular class.
  # So, let's get the index of the highest energy:

  _, predicted = torch.max(outputs, 1)

  print('Predicted: ', ' '.join('%5s' % classes[predicted[j]]
                for j in range(4)))

  ########################################################################
  # The results seem pretty good.
  #
  # Let us look at how the network performs on the whole dataset.

  correct = 0
  total = 0
  with torch.no_grad():
    for data in testloader:

      # if(torch.cuda.is_available()):
      #   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      #   images, labels = data[0].to(device), data[1].to(device)
      # else:
      #   images, labels = data

      images, labels = data
      outputs = net(images)
      _, predicted = torch.max(outputs.data, 1)
      total += labels.size(0)
      correct += (predicted == labels).sum().item()

  print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

  ########################################################################
  # That looks way better than chance, which is 10% accuracy (randomly picking
  # a class out of 10 classes).
  # Seems like the network learnt something.
  #
  # Hmmm, what are the classes that performed well, and the classes that did
  # not perform well:

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
    print('Accuracy of %5s : %2d %%' % (
      classes[i], 100 * class_correct[i] / class_total[i]))

########################################################################
# Okay, so what next?
#
# How do we run these neural networks on the GPU?
#
# Training on GPU
# ----------------
# Just like how you transfer a Tensor onto the GPU, you transfer the neural
# net onto the GPU.
#
# Let's first define our device as the first visible cuda device if we have
# CUDA available:

########################################################################
# The rest of this section assumes that ``device`` is a CUDA device.
#
# Then these methods will recursively go over all modules and convert their
# parameters and buffers to CUDA tensors:
#
# .. code:: python
#
#     net.to(device)
#
#
# Remember that you will have to send the inputs and targets at every step
# to the GPU too:
#
# .. code:: python
#
#         inputs, labels = data[0].to(device), data[1].to(device)
#
# Why dont I notice MASSIVE speedup compared to CPU? Because your network
# is really small.
#
# **Exercise:** Try increasing the width of your network (argument 2 of
# the first ``nn.Conv2d``, and argument 1 of the second ``nn.Conv2d`` –
# they need to be the same number), see what kind of speedup you get.
#
# **Goals achieved**:
#
# - Understanding PyTorch's Tensor library and neural networks at a high level.
# - Train a small neural network to classify images
#
# Training on multiple GPUs
# -------------------------
# If you want to see even more MASSIVE speedup using all of your GPUs,
# please check out :doc:`data_parallel_tutorial`.
#
# Where do I go next?
# -------------------
#
# -  :doc:`Train neural nets to play video games </intermediate/reinforcement_q_learning>`
# -  `Train a state-of-the-art ResNet network on imagenet`_
# -  `Train a face generator using Generative Adversarial Networks`_
# -  `Train a word-level language model using Recurrent LSTM networks`_
# -  `More examples`_
# -  `More tutorials`_
# -  `Discuss PyTorch on the Forums`_
# -  `Chat with other users on Slack`_
#
# .. _Train a state-of-the-art ResNet network on imagenet: https://github.com/pytorch/examples/tree/master/imagenet
# .. _Train a face generator using Generative Adversarial Networks: https://github.com/pytorch/examples/tree/master/dcgan
# .. _Train a word-level language model using Recurrent LSTM networks: https://github.com/pytorch/examples/tree/master/word_language_model
# .. _More examples: https://github.com/pytorch/examples
# .. _More tutorials: https://github.com/pytorch/tutorials
# .. _Discuss PyTorch on the Forums: https://discuss.pytorch.org/
# .. _Chat with other users on Slack: https://pytorch.slack.com/messages/beginner/

  del dataiter
  _ = input("Press key to finish..")


if __name__ == '__main__':
  main()


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
