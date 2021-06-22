import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel import BaseModel

# ======================================================== #
class AE(nn.Module):
  def __init__(self, encoder_name, decoder_name, z_size):
    super(AE, self).__init__()
    self.encoder = Encoder(encoder_name, z_size)
    self.decoder = Decoder(decoder_name, z_size)

  def forward(self, x):
    z = self.encoder(x)
    x = self.decoder(z)
    return z, x

class Encoder(BaseModel):
  
  def __init__(self, model_name, z_size):
    super(Encoder, self).__init__("Encoder", model_name)
    self.z_size = z_size
    self.set_model()

  def set_model_1(self):
    self.conv1 = nn.Conv2d(1, 8, kernel_size=(5,5))
    self.bn1 = nn.BatchNorm2d(8)
    self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,5))
    self.bn2 = nn.BatchNorm2d(16)
    self.conv3 = nn.Conv2d(16, 32, kernel_size=(5,5))
    self.bn3 = nn.BatchNorm2d(32)
    self.fc1 = nn.Linear(32*16*16, self.z_size)

  def forward_1(self, x):
    x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
    x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
    x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
    x = x.view(-1, 32*16*16)
    x = torch.sigmoid(self.fc1(x))
    return x

class Decoder(BaseModel):
  
  def __init__(self, model_name, z_size):
    super(Decoder, self).__init__("Decoder", model_name)
    self.z_size = z_size
    self.set_model()

  def set_model_1(self):
    self.fc1 = nn.Linear(self.z_size, 32 * 16 * 16)
    self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(5,5))
    self.conv3 = nn.ConvTranspose2d(16, 8, kernel_size=(5,5))
    self.conv4 = nn.ConvTranspose2d(8, 1, kernel_size=(5,5))

  def forward_1(self, x):
    x = F.relu(self.fc1(x))
    x = x.view(-1, 32, 16, 16)
    x = F.leaky_relu(self.conv2(x), 0.2)
    x = F.leaky_relu(self.conv3(x), 0.2)
    x = F.leaky_relu(self.conv4(x), 0.2)
    x = torch.tanh(x)
    return x