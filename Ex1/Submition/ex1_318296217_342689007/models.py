import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import enum

# Using enum class create enumerations
class Net(nn.Module):

  def __init__(self, args):
    super(Net, self).__init__()
    
    self.set_mappings() # sets the mapping of the model type to the relevant functions

    # if cuda is available define the net to be cuda
    if(torch.cuda.is_available()):
      device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
      self.to(device)

    self.model_name = ModelType.parse(args.model_name) # sets the enum of the model as a public field
    self.set_model() # sets the parameters of the net according to the model

    self.lr = args.lr # sets the lr public field
    self.momentum = args.momentum # sets the momentum public field
    self.loss_function = nn.CrossEntropyLoss() # sets the loss function as a public field
    if(torch.cuda.is_available()):
      self.loss_function .cuda() # if cuda is available enable cuda to the loss function 
    self.optimizer = optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum) # sets the optimizer as a public field

  '''sets the model depending on the running model type'''
  def set_model(self):
    print(f"Running model: {self.model_name}")

    # if the mapping exist, run it. else throw an exception
    if(self.model_name in self._set_model_mapping):
      return self._set_model_mapping[self.model_name]()
    else:
      raise Exception(f"No such model named: {self.model_name}")

  '''forwards depending on the running model type'''
  def forward(self, x):
    # if the mapping exist, run it. else throw an exception
    if(self.model_name in self._forward_mapping):
      return self._forward_mapping[self.model_name](x)
    else:
      raise Exception(f"No such model named: {self.model_name}")

  # define the mapping for the different models
  '''defines switching of the models to the correct functions'''
  def set_mappings(self):
    self._set_model_mapping = {
      ModelType.ORIGINAL: self.set_model_Original,
      ModelType.Q1_1: self.set_model_Q1_1,
      ModelType.Q1_2: self.set_model_Q1_2,
      ModelType.Q1_3: self.set_model_Q1_3,
      ModelType.Q1_4: self.set_model_Q1_4,
      ModelType.Q1_5: self.set_model_Q1_5,
      ModelType.Q1_6: self.set_model_Q1_6,
      ModelType.Q1_7: self.set_model_Q1_7,
      ModelType.Q1_8: self.set_model_Q1_8,
      ModelType.Q1_9: self.set_model_Q1_9,
      ModelType.Q1_10: self.set_model_Q1_10,
      ModelType.Q1_11: self.set_model_Q1_11,
      ModelType.Q1_12: self.set_model_Q1_12,
      ModelType.Q1_13: self.set_model_Q1_13,
      ModelType.Q1_14: self.set_model_Q1_14,
      ModelType.Q1_15: self.set_model_Q1_15,
      ModelType.Q1_16: self.set_model_Q1_16,
      ModelType.Q1_17: self.set_model_Q1_17,
      ModelType.Q1_18: self.set_model_Q1_18,
      ModelType.Q1_19: self.set_model_Q1_19,
      ModelType.Q1_20: self.set_model_Q1_20,
      ModelType.Q1_21: self.set_model_Q1_21,
      ModelType.Q1_22: self.set_model_Q1_22,
      ModelType.Q1_23: self.set_model_Q1_23,
      ModelType.Q1_24: self.set_model_Q1_24,
      ModelType.Q1_25: self.set_model_Q1_25,
      ModelType.Q2_1: self.set_model_Q2_1,
      ModelType.Q2_2: self.set_model_Q2_2,
    }
    self._forward_mapping = {
      ModelType.ORIGINAL: self.forward_Original,
      ModelType.Q1_1: self.forward_Q1_1,
      ModelType.Q1_2: self.forward_Q1_2,
      ModelType.Q1_3: self.forward_Q1_3,
      ModelType.Q1_4: self.forward_Q1_4,
      ModelType.Q1_5: self.forward_Q1_5,
      ModelType.Q1_6: self.forward_Q1_6,
      ModelType.Q1_7: self.forward_Q1_7,
      ModelType.Q1_8: self.forward_Q1_8,
      ModelType.Q1_9: self.forward_Q1_9,
      ModelType.Q1_10: self.forward_Q1_10,
      ModelType.Q1_11: self.forward_Q1_11,
      ModelType.Q1_12: self.forward_Q1_12,
      ModelType.Q1_13: self.forward_Q1_13,
      ModelType.Q1_14: self.forward_Q1_14,
      ModelType.Q1_15: self.forward_Q1_15,
      ModelType.Q1_16: self.forward_Q1_16,
      ModelType.Q1_17: self.forward_Q1_17,
      ModelType.Q1_18: self.forward_Q1_18,
      ModelType.Q1_19: self.forward_Q1_19,
      ModelType.Q1_20: self.forward_Q1_20,
      ModelType.Q1_21: self.forward_Q1_21,
      ModelType.Q1_22: self.forward_Q1_22,
      ModelType.Q1_23: self.forward_Q1_23,
      ModelType.Q1_24: self.forward_Q1_24,
      ModelType.Q1_25: self.forward_Q1_25,
      ModelType.Q2_1: self.forward_Q2_1,
      ModelType.Q2_2: self.forward_Q2_2,
    }

# ============== Original ============== #

  ''' this is the original net given '''

  def set_model_Original(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Original(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_1 ============== #

  ''' this is the original net given '''

  def set_model_Q1_1(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_1(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_2 ============== #

  ''' in this net we removed a fully connected later '''

  def set_model_Q1_2(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 84)
    self.fc2 = nn.Linear(84, 10)

  def forward_Q1_2(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_3 ============== #

  ''' in this net we changed the conv layers to have a kernel size of 3 instead of 5 '''

  def set_model_Q1_3(self):
    self.conv1 = nn.Conv2d(3, 6, 3)
    self.conv2 = nn.Conv2d(6, 16, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 6 * 6, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_3(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 6 * 6) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_4 ============== #

  ''' in this net we added 3 more layers of convolution, and since the output was small, 
      we removed a fully connected '''

  def set_model_Q1_4(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 6, 5)
    self.conv3 = nn.Conv2d(6, 6, 5)
    self.conv4 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 2 * 2, 20)
    self.fc2 = nn.Linear(20, 10)

  def forward_Q1_4(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool(x)
    x = x.view(-1, 16 * 2 * 2) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_5 ============== #

  ''' to have more neurons, instead of having conv layers with kernel 5x5, we have conv layers of kernel 3x3. 
      this way we have 4 layers of conv (where applaying pool after every two conv layers)'''

  def set_model_Q1_5(self):
    self.conv1 = nn.Conv2d(3, 8, 3)
    self.conv2 = nn.Conv2d(8, 8, 3)
    self.conv3 = nn.Conv2d(8, 8, 3)
    self.conv4 = nn.Conv2d(8, 16, 3)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_5(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool(x)
    x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_6 ============== #

  ''' in this net we lowered the conv kernel to 3 but also added padding to the layers 
      so it doesn't get smaller, but do still applay pooling after each one '''

  def set_model_Q1_6(self):
    self.conv1 = nn.Conv2d(3, 6, 3, padding=1)
    self.conv2 = nn.Conv2d(6, 10, 3, padding=1)
    self.conv3 = nn.Conv2d(10, 16, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 4 * 4, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_6(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = x.view(-1, 16 * 4 * 4) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_7 ============== #

  ''' this net is just like net Q1_2 but with more neurons in the fully connected layers '''

  def set_model_Q1_7(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(16 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 10)

  def forward_Q1_7(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 16 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_8 ============== #

  ''' in this net we added channels to the conv layers (from 6 to 8 in conv1 and from 16 to 20 in conv2) 
      resulting in a more complex net '''

  def set_model_Q1_8(self):
    self.conv1 = nn.Conv2d(3, 8, 5)
    self.conv2 = nn.Conv2d(8, 20, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(20 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_8(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 20 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_9 ============== #

  ''' in this net we added channels to the conv layers (from 6 to 10 in conv1 and from 16 to 24 in conv2) 
      resulting in a more complex net '''

  def set_model_Q1_9(self):
    self.conv1 = nn.Conv2d(3, 10, 5)
    self.conv2 = nn.Conv2d(10, 24, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(24 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_9(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 24 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_10 ============== #

  ''' in this net we added channels to the conv layers (from 6 to 10 in conv1 and from 16 to 28 in conv2) 
      resulting in a more complex net '''

  def set_model_Q1_10(self):
    self.conv1 = nn.Conv2d(3, 10, 5)
    self.conv2 = nn.Conv2d(10, 28, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(28 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_10(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 28 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_11 ============== #

  ''' this net is designed to overfit by having 36 channels after the convolution layer which result 
      in many neurons in the fully connected layers as well as an additional fully connected layer'''

  def set_model_Q1_11(self):
    self.conv1 = nn.Conv2d(3, 10, 5)
    self.conv2 = nn.Conv2d(10, 36, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(36 * 5 * 5, 400)
    self.fc2 = nn.Linear(400, 120)
    self.fc3 = nn.Linear(120, 84)
    self.fc4 = nn.Linear(84, 10)

  def forward_Q1_11(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 36 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = F.relu(self.fc3(x))
    x = self.fc4(x)
    return x

# ============== Q1_12 ============== #

  ''' in continuation of Q1_4: Q1_4 was underfit due to so few neurons. 
      Therefore here we want to increase the number while leaving the conv layers. 
      Hence we will add padding to the conv layers but remove some of the channels'''

  def set_model_Q1_12(self):
    self.conv1 = nn.Conv2d(3, 6, 5, padding=1)
    self.conv2 = nn.Conv2d(6, 6, 5, padding=1)
    self.conv3 = nn.Conv2d(6, 6, 5, padding=1)
    self.conv4 = nn.Conv2d(6, 10, 5, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(10 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_12(self, x):
    x = F.relu(self.conv1(x))
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = F.relu(self.conv3(x))
    x = F.relu(self.conv4(x))
    x = self.pool(x)
    x = x.view(-1, 10 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_13 ============== #
  
  ''' to try to avoid overfitting, here we simply lower the number of channels of the last conv layer to 8'''
  
  def set_model_Q1_13(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 8, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(8 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_13(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 8 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_14 ============== #
  
  ''' to try to avoid overfitting, here we simply lower the number of channels of the last conv layer to 10'''
  
  def set_model_Q1_14(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 10, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(10 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_14(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 10 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_15 ============== #
  
  ''' to try to avoid overfitting, here we simply lower the number of channels of the last conv layer to 12'''
  
  def set_model_Q1_15(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 12, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(12 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_15(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 12 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_16 ============== #
  
  ''' to try to avoid overfitting, here we simply lower the number of channels of the last conv layer to 14'''
  
  def set_model_Q1_16(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 14, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(14 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_16(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(x)
    x = F.relu(self.conv2(x))
    x = self.pool(x)
    x = x.view(-1, 14 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_17 ============== #
  
  ''' obviously we still have overfitting. So changed the conv layer to max of 6 channels and lowering the fully connected respectively
      Resulting in 30,044 parameters to learn.'''

  def set_model_Q1_17(self):
    self.conv1 = nn.Conv2d(3, 4, 5)
    self.conv2 = nn.Conv2d(4, 6, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(6 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_17(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 6 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_18 ============== #
  
  ''' obviously we still have overfitting. So changed the conv layer to max of 5 channels and lowering the fully connected respectively
      Total params: 26,943'''

  def set_model_Q1_18(self):
    self.conv1 = nn.Conv2d(3, 4, 5)
    self.conv2 = nn.Conv2d(4, 5, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(5 * 5 * 5, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q1_18(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 5 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.fc3(x)
    return x

# ============== Q1_19 ============== #
  
  '''We now change the approach. Instead of lowering the number of neurons, maybe the right thing is to 
    have more convolutional parameters as opposed to fully connected ones? Therefore, we added many 
    convolutional layers which bump it up to 128 channels! This indeed gave an incredible result 
    meaning we are getting closer. Total params: 550,570 '''

  def set_model_Q1_19(self):
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
    self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv6 = nn.Conv2d(128, 128, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(128 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward_Q1_19(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = self.pool(F.relu(self.conv4(x)))
    x = F.relu(self.conv5(x))
    x = self.pool(F.relu(self.conv6(x)))
    x = x.view(-1, 128 * 4 * 4) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_20 ============== #
  
  '''Continuing the idea of Q1_19, but with less parameters - 356,810'''

  def set_model_Q1_20(self):
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(128 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward_Q1_20(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = self.pool(F.relu(self.conv3(x)))
    x = x.view(-1, 128 * 4 * 4) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_21 ============== #
  
  '''Continuing the idea of Q1_19, but with less parameters - 164,234'''

  def set_model_Q1_21(self):
    self.conv1 = nn.Conv2d(3, 32, 3)
    self.conv2 = nn.Conv2d(32, 32, 3)
    self.conv3 = nn.Conv2d(32, 64, 3)
    self.conv4 = nn.Conv2d(64, 64, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 4 * 4, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward_Q1_21(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = self.pool(F.relu(self.conv4(x)))
    x = x.view(-1, 64 * 4 * 4) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_22 ============== #

  '''Continuing the idea of Q1_19, but with less parameters - 105,258'''
  
  def set_model_Q1_22(self):
    self.conv1 = nn.Conv2d(3, 32, 5)
    self.conv2 = nn.Conv2d(32, 64, 5)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(64 * 5 * 5, 32)
    self.fc2 = nn.Linear(32, 10)

  def forward_Q1_22(self, x):
    x = self.pool(F.relu(self.conv1(x)))
    x = self.pool(F.relu(self.conv2(x)))
    x = x.view(-1, 64 * 5 * 5) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_23 ============== #
  
  '''Continuing the idea of Q1_19, but with more parameters - 960,298'''

  def set_model_Q1_23(self):
    self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
    self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
    self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
    self.conv4 = nn.Conv2d(64, 64, 3, padding=1)
    self.conv5 = nn.Conv2d(64, 128, 3, padding=1)
    self.conv6 = nn.Conv2d(128, 256, 3, padding=1)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(256 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward_Q1_23(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = self.pool(F.relu(self.conv4(x)))
    x = F.relu(self.conv5(x))
    x = self.pool(F.relu(self.conv6(x)))
    x = x.view(-1, 256 * 4 * 4) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_24 ============== #
  
  '''Continuing the idea of Q1_19, but with more parameters - 1,060,010'''

  def set_model_Q1_24(self):
    self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
    self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
    self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
    self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
    self.conv5 = nn.Conv2d(64, 128, 5, padding=2)
    self.conv6 = nn.Conv2d(128, 128, 5, padding=2)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(128 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward_Q1_24(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = self.pool(F.relu(self.conv4(x)))
    x = F.relu(self.conv5(x))
    x = self.pool(F.relu(self.conv6(x)))
    x = x.view(-1, 128 * 4 * 4) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q1_25 ============== #
  
  '''Continuing the idea of Q1_19, but with more parameters - 2,141,610'''

  def set_model_Q1_25(self):
    self.conv1 = nn.Conv2d(3, 32, 5, padding=2)
    self.conv2 = nn.Conv2d(32, 32, 5, padding=2)
    self.conv3 = nn.Conv2d(32, 64, 5, padding=2)
    self.conv4 = nn.Conv2d(64, 64, 5, padding=2)
    self.conv5 = nn.Conv2d(64, 128, 5, padding=2)
    self.conv6 = nn.Conv2d(128, 128, 5, padding=2)
    self.conv7 = nn.Conv2d(128, 256, 5, padding=2)
    self.pool = nn.MaxPool2d(2, 2)
    self.fc1 = nn.Linear(256 * 4 * 4, 128)
    self.fc2 = nn.Linear(128, 10)

  def forward_Q1_25(self, x):
    x = F.relu(self.conv1(x))
    x = self.pool(F.relu(self.conv2(x)))
    x = F.relu(self.conv3(x))
    x = self.pool(F.relu(self.conv4(x)))
    x = F.relu(self.conv5(x))
    x = F.relu(self.conv6(x))
    x = self.pool(F.relu(self.conv7(x)))
    x = x.view(-1, 256 * 4 * 4) # reshapes for the fully connected
    x = F.relu(self.fc1(x))
    x = self.fc2(x)
    return x

# ============== Q2_1 ============== # 

# Q2 - delete the non-linear layers
  def set_model_Q2_1(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 24 * 24, 120)
    self.fc2 = nn.Linear(120, 84)
    self.fc3 = nn.Linear(84, 10)

  def forward_Q2_1(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = x.view(-1, 16 * 24 * 24) # reshapes for the fully connected
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc3(x)
      return x

# ============== Q2_2 ============== # 

# Q2 - delete the non-linear layers
  def set_model_Q2_2(self):
    self.conv1 = nn.Conv2d(3, 6, 5)
    self.conv2 = nn.Conv2d(6, 16, 5)
    self.fc1 = nn.Linear(16 * 24 * 24, 280)
    self.fc2 = nn.Linear(280, 120)
    self.fc3 = nn.Linear(120, 10)

  def forward_Q2_2(self, x):
      x = self.conv1(x)
      x = self.conv2(x)
      x = x.view(-1, 16 * 24 * 24) # reshapes for the fully connected
      x = self.fc1(x)
      x = self.fc2(x)
      x = self.fc3(x)
      return x

# -------------------------------- #

  # Helper function to debug the shape of the tensor in runtime
  def debug_shape(self, x, msg, print_flag):
    if(print_flag):
      print(str(x.shape) + " " + msg)

# ================================ # 
# ================================ # 

# Enum class to choose the running model
class ModelType(enum.Enum):
  ORIGINAL = 1
  Q1_1 = 2
  Q1_2 = 3
  Q1_3 = 4
  Q1_4 = 5
  Q1_5 = 6
  Q1_6 = 7
  Q1_7 = 8
  Q1_8 = 9
  Q1_9 = 10
  Q1_10 = 11
  Q1_11 = 12
  Q1_12 = 13
  Q1_13 = 14
  Q1_14 = 15
  Q1_15 = 16
  Q1_16 = 17
  Q1_17 = 18
  Q1_18 = 19
  Q1_19 = 20
  Q1_20 = 21
  Q1_21 = 22
  Q1_22 = 23
  Q1_23 = 24
  Q1_24 = 25
  Q1_25 = 26
  Q2_1 = 27
  Q2_2 = 28

  @staticmethod
  # parse a string name and get the corresponding enum
  def parse(model_name):
    models_dict = {
        "ORIGINAL".lower(): ModelType.ORIGINAL,
        "Q1_1".lower(): ModelType.Q1_1,
        "Q1_2".lower(): ModelType.Q1_2,
        "Q1_3".lower(): ModelType.Q1_3,
        "Q1_4".lower(): ModelType.Q1_4,
        "Q1_5".lower(): ModelType.Q1_5,
        "Q1_6".lower(): ModelType.Q1_6,
        "Q1_7".lower(): ModelType.Q1_7,
        "Q1_8".lower(): ModelType.Q1_8,
        "Q1_9".lower(): ModelType.Q1_9,
        "Q1_10".lower(): ModelType.Q1_10,
        "Q1_11".lower(): ModelType.Q1_11,
        "Q1_12".lower(): ModelType.Q1_12,
        "Q1_13".lower(): ModelType.Q1_13,
        "Q1_14".lower(): ModelType.Q1_14,
        "Q1_15".lower(): ModelType.Q1_15,
        "Q1_16".lower(): ModelType.Q1_16,
        "Q1_17".lower(): ModelType.Q1_17,
        "Q1_18".lower(): ModelType.Q1_18,
        "Q1_19".lower(): ModelType.Q1_19,
        "Q1_20".lower(): ModelType.Q1_20,
        "Q1_21".lower(): ModelType.Q1_21,
        "Q1_22".lower(): ModelType.Q1_22,
        "Q1_23".lower(): ModelType.Q1_23,
        "Q1_24".lower(): ModelType.Q1_24,
        "Q1_25".lower(): ModelType.Q1_25,
        "Q2_1".lower(): ModelType.Q2_1,
        "Q2_2".lower(): ModelType.Q2_2,
    }

    if(model_name.lower() in models_dict):
      return models_dict[model_name.lower()]
    else:
      raise Exception(f"No such model named: {model_name}")