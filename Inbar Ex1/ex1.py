import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

"""initializtion of datasets and global parameters"""
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Net(nn.Module):
    """Class Net, based on the vanilla net architecture mentioned in the exercise pdf.
       The constructor receives 2 parameters: conv_filter_num which allows modifiying
       the number of channels in the second convolution layer, and  fc_fiter_num
       which allows to choose the number of neurons in the first fully-connected layer.
       In the report I submitted, I got the results I needed by manipulation of the second
       convolution layer only.
    """
    filter_num = 16

    def __init__(self, conv_filter_num, fc_fiter_num):
        super(Net, self).__init__()
        self.filter_num = conv_filter_num
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, conv_filter_num, 5)
        self.fc1 = nn.Linear(conv_filter_num * 5 * 5, fc_fiter_num)
        self.fc2 = nn.Linear(fc_fiter_num, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.filter_num * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LinearNet(nn.Module):
    """Linear network. based on the vanilla network mentioned in the exercise, after removing all
        non-linear components from it. """
    def __init__(self, fc_fiter_num):
        super(LinearNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.AvgPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, fc_fiter_num)
        self.fc3 = nn.Linear(fc_fiter_num, 10)

    def forward(self, x):
        x = self.pool(self.conv1(x))
        x = self.pool(self.conv2(x))
        x = x.view(-1, 16 * 5 * 5)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x


def train(net, epochs, name, permutation = None):
    """trains the network. Receives a network object, number of epochs, name
        (in order to save the trained model under a specific name) and an optional parameter called 'permutation'
        if permutation is None, trains the networks as usual with the number of epochs received. otherwise, if
        permutation == 'permutation', a permutation is randomly picked for every batch and all the images in the batch
         are permutated with it. otherwise, if permutation is an actual permutation, permutatates all batches with it
        """
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    running_loss = 0.0
    total = 0
    for epoch in range(epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        total = 0
        for i, data in enumerate(trainloader,0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            if permutation is not None:
                if permutation=="permutation":
                    inputs = apply_permutaion(np.random.permutation(1024), inputs)
                else:
                    inputs = apply_permutaion(permutation, inputs)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            total+=labels.size(0)
    PATH = "./cifar_net_"+name+".pth"
    torch.save(net.state_dict(), PATH)
    print('Finished Training')
    return PATH, running_loss*4/total


def test(path, net, permutation = None):
    """ tests the network on the rest set. receives path of traind network, network object template, and an optional
        parameter called permutation.  The logic is the same as described in train function documentation."""
    criterion = nn.CrossEntropyLoss()
    net.load_state_dict(torch.load(path))
    correct = 0
    total = 0
    total_clsses = [0 for i in range(10)]
    success_clsses = [0 for i in range(10)]
    with torch.no_grad():
        running_loss = 0.0
        for i, data in enumerate(testloader):
            images, labels = data
            if permutation is not None:
                if permutation == "permutation":
                    images = apply_permutaion(np.random.permutation(1024), images)
                else:
                    images = apply_permutaion(permutation, images)
            outputs = net(images)
            loss = criterion(outputs, labels)
            # print statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            c = (predicted==labels).squeeze()
            for j in range(4):
                lable = labels[j]
                success_clsses[lable] += c[j].item()
                total_clsses[lable] += 1

    print('Accuracy of the network on the 10000 test images: %d %%' % (
            100 * correct / total))
    for i in range(10):
        print("class: "+classes[i]+" success rate: "+str(success_clsses[i]*100/total_clsses[i])+"%")
    return running_loss*4/total


def train_test(conv_filt_num, fc_filt_num, linear=False):
    """ helper function, which conducts tha all flow of training and testing the net. I used it for manipulating
        networks' parameters easily"""
    if linear:
        net = LinearNet(fc_filt_num)
    else:
        net = Net(conv_filt_num, fc_filt_num)
    count = 244 + (conv_filt_num*25) + fc_filt_num
    path, train_loss = train(net,7, str(count)+"_net")
    test_loss = test(path, net)
    return count, train_loss, test_loss


def manipulate_conv_layer():
    """ The function used for creating practical question 1 plot."""
    parameters = [16,64,256,512,1024]
    count_list = list()
    train_losses = list()
    test_losses = list()
    for par in parameters:
        count, train_loss, test_loss = train_test(par, 120)
        count_list.append(count)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.title('loss')
    plt.plot(count_list, train_losses), plt.plot(count_list, test_losses)
    plt.legend(['train', 'test'])
    plt.xlabel("#neurons")
    plt.ylabel("average loss")
    plt.show()


def test_linear_model():
    """ The function used for creating practical question 2 plot"""
    parameters = [84,360,600,1200,2400]
    count_list = list()
    train_losses = list()
    test_losses = list()
    for par in parameters:                                              
        count, train_loss, test_loss = train_test(16, par, True)
        count_list.append(count)
        train_losses.append(train_loss)
        test_losses.append(test_loss)

    plt.title('loss')
    plt.plot(count_list, train_losses), plt.plot(count_list, test_losses)
    plt.legend(['train', 'test'])
    plt.xlabel("#neurons")
    plt.ylabel("average loss")
    plt.show()


def apply_permutaion(permutation,batch):
    """ receives a fixed size batch of shape (4,3,32,32) and a permutaion over the shape (1024),
        reshapes the images in the batch, applies the given permutation on each and then reshapes it again to the former
        shape and returns the results"""
    tmp = batch.reshape(4,3,1024)
    r,c = np.meshgrid(permutation, np.arange(0,3))
    res = tmp[:,c,r]
    return res.reshape(4,3,32,32)


def locality_check():
    """ The function used for answering practical question 3 """
    permutation = np.random.permutation(1024)
    net = Net(16,120)
    count = 244 + (16*25) + 120
    path, train_loss = train(net,5, str(count)+"_net", permutation)
    test_loss = test(path, net, permutation)
    print(test_loss)


def spatial_check():
    """ The function used for answering practical question 4"""
    net = Net(16, 120)
    count = 244 + (16*25) + 120
    path, train_loss = train(net,5, str(count)+"_net", "permutation")
    test_loss = test(path, net, "permutation")
    print(test_loss)

