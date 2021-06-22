import torch
import torch.nn as nn
import torch.nn.functional as F

from basemodel import BaseModel

# ======================================================== #
# ==================== Loss Functions ==================== #
# ======================================================== #

class NonSaturatedCrossEntropy(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight=None) -> None:
        super(NonSaturatedCrossEntropy, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return super(NonSaturatedCrossEntropy, self).forward(input, (target+1) % 2)
    def __str__(self):
        return "NonSaturated"

class CrossEntropy(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight=None) -> None:
        super(CrossEntropy, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return -super(CrossEntropy, self).forward(input, target)

    def __str__(self):
        return "CrossEntropy"


class BCEWithLogitsLoss(nn.BCEWithLogitsLoss):
    def __init__(self, weight=None, size_average=None, reduce=None, reduction: str = 'mean', pos_weight=None) -> None:
        super(BCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction, pos_weight)
        self.register_buffer('weight', weight)
        self.register_buffer('pos_weight', pos_weight)

    def forward(self, input, target):
        return super(BCEWithLogitsLoss, self).forward(input, target)

    def __str__(self):
        return "CrossEntropy"

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
    # initializers
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
        x = self.fc1(x)
        return x

    def set_model_2(self):
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, self.z_size)

    def forward_2(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        return x

## FROM HER ON MODELS ARE FOR THE CELEBS DATASET!! ##

    def set_model_3(self):
        times = 8
        self.conv1 = nn.Conv2d(3, 1*times, kernel_size=(4,4), padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(1*times)
        self.conv2 = nn.Conv2d(1*times, 2*times, kernel_size=(4,4), padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(2*times)
        self.conv3 = nn.Conv2d(2*times, 4*times, kernel_size=(4,4), padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(4*times)
        self.conv4 = nn.Conv2d(4*times, 8*times, kernel_size=(4,4), padding=2, stride=2)
        self.bn4 = nn.BatchNorm2d(8*times)
        self.view_out = 8*times*5*5
        self.fc1 = nn.Linear(self.view_out, self.z_size)

    def forward_3(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = x.view(-1, self.view_out)
        x = self.fc1(x)
        return x

    def set_model_4(self):
        times = 8
        self.conv1 = nn.Conv2d(3, 1*times, kernel_size=(4,4), padding=2, stride=2)
        self.bn1 = nn.BatchNorm2d(1*times)
        self.conv2 = nn.Conv2d(1*times, 2*times, kernel_size=(4,4), padding=2, stride=2)
        self.bn2 = nn.BatchNorm2d(2*times)
        self.conv3 = nn.Conv2d(2*times, 4*times, kernel_size=(4,4), padding=2, stride=2)
        self.bn3 = nn.BatchNorm2d(4*times)
        self.view_out = 4*times*9*9
        self.fc1 = nn.Linear(self.view_out, self.z_size)

    def forward_4(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = x.view(-1, self.view_out)
        x = self.fc1(x)
        return x

    def set_model_5(self):
        self.conv1 = nn.Conv2d(3, 8, kernel_size=(5,5))
        self.bn1 = nn.BatchNorm2d(8)
        self.conv2 = nn.Conv2d(8, 16, kernel_size=(5,5))
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(5,5))
        self.bn3 = nn.BatchNorm2d(32)
        self.conv4 = nn.Conv2d(32, 64, kernel_size=(5,5), stride = 2)
        self.bn4 = nn.BatchNorm2d(64)        
        self.fc1 = nn.Linear(64*24*24, self.z_size)

    def forward_5(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn4(self.conv4(x)), 0.2)
        x = x.view(-1, 64*24*24)
        # x = x.view(-1, 32*16*16)
        x = self.fc1(x)
        return x

class Decoder(BaseModel):
    # initializers
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

    def set_model_2(self):
        self.fc1 = nn.Linear(self.z_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 512)
        self.fc4 = nn.Linear(512, 784)

    def forward_2(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = x.view(-1,1, 28, 28)
        x = torch.tanh(x)
        return x

## FROM HER ON MODELS ARE FOR THE CELEBS DATASET!! ##

    def set_model_3(self):
        times = 8
        self.view_ch = 8*times  
        self.fc1 = nn.Linear(self.z_size, self.view_ch*8*8)
        self.conv1 = nn.ConvTranspose2d(8*times, 4*times, kernel_size=(4,4), padding=1, stride=2)
        self.conv2 = nn.ConvTranspose2d(4*times, 2*times, kernel_size=(4,4), padding=1, stride=2)
        self.conv3 = nn.ConvTranspose2d(2*times, 1*times, kernel_size=(4,4), padding=1, stride=2)
        self.conv4 = nn.ConvTranspose2d(1*times, 3, kernel_size=(4,4), padding=1, stride=2)

    def forward_3(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, self.view_ch, 8, 8)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.tanh(x)
        return x

    def set_model_4(self):
        times = 8
        self.view_ch = 4*times  
        self.fc1 = nn.Linear(self.z_size, self.view_ch*8*8)
        self.conv1 = nn.ConvTranspose2d(4*times, 2*times, kernel_size=(4,4), padding=1, stride=2)
        self.conv2 = nn.ConvTranspose2d(2*times, 1*times, kernel_size=(4,4), padding=1, stride=2)
        self.conv3 = nn.ConvTranspose2d(1*times, 3, kernel_size=(4,4), padding=1, stride=2)

    def forward_4(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, self.view_ch, 8, 8)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = torch.tanh(x)
        return x

    def set_model_5(self):
        self.fc1 = nn.Linear(self.z_size, 64 * 24 * 24)
        self.conv1 = nn.ConvTranspose2d(64, 32, kernel_size=(5,5), stride=2, output_padding=1)
        self.conv2 = nn.ConvTranspose2d(32, 16, kernel_size=(5,5))
        self.conv3 = nn.ConvTranspose2d(16, 8, kernel_size=(5,5))
        self.conv4 = nn.ConvTranspose2d(8, 3, kernel_size=(5,5))

    def forward_5(self, x):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 64, 24, 24)
        x = F.leaky_relu(self.conv1(x), 0.2)
        x = F.leaky_relu(self.conv2(x), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = F.leaky_relu(self.conv4(x), 0.2)
        x = torch.tanh(x)
        return x

# ======================================================== #
# ======================== Celebs ======================== #
# ======================================================== #

class Generator_celeb(BaseModel):
    # initializers
    def __init__(self, model_name, input_size, d):
        super(Generator_celeb, self).__init__("Generator_celeb", model_name)
        self.d = d
        self.input_size = input_size
        self.set_model()

# ============== Celebs: Generator Model_1 ============== #

    def set_model_1(self):
        self.fc1 = nn.Linear(self.input_size, self.d * 8 * 4 * 4)
        self.transpose_conv1 = nn.ConvTranspose2d(self.d * 8, self.d * 4, 4,stride = 2, padding = 1)
        self.bn1 = nn.BatchNorm2d(self.d*4)
        self.transposed_conv2 = nn.ConvTranspose2d(self.d*4, self.d*2, 4, stride = 2,  padding = 1)
        self.bn2 = nn.BatchNorm2d(self.d*2)
        self.transposed_conv3 = nn.ConvTranspose2d(self.d*2, self.d, 4, stride = 2,  padding = 1)
        self.bn3 = nn.BatchNorm2d(self.d)
        self.transposed_conv4 = nn.ConvTranspose2d(self.d, 3, 4, stride=2,  padding = 1)

    def forward_1(self, input):
        # x = F.leaky_relu(self.fc1(input), 0.2)
        # x = x.view(-1, self.d * 8, 4, 4)
        # x = F.leaky_relu(self.bn1(self.transpose_conv1(x)))
        # x = F.leaky_relu(self.bn2(self.transposed_conv2(x)))
        # x = F.leaky_relu(self.bn3(self.transposed_conv3(x)))
        # x = torch.tanh(self.transposed_conv4(x)) # do not use sigmoid! only tanh

        x = F.relu(self.fc1(input))
        x = x.view(-1, self.d * 8, 4, 4)
        x = F.relu(self.bn1(self.transpose_conv1(x)))
        x = F.relu(self.bn2(self.transposed_conv2(x)))
        x = F.relu(self.bn3(self.transposed_conv3(x)))
        x = torch.tanh(self.transposed_conv4(x)) # do not use sigmoid! only tanh
        return x
        
# ============== Celebs: Generator Model_2 ============== #

    def set_model_2(self):
        self.fc1 = nn.Linear(self.input_size, 512)
        self.fc2 = nn.Linear(self.fc1.out_features, 1024)
        self.fc3 = nn.Linear(self.fc2.out_features, 2704)
        self.transposed_conv1 = nn.ConvTranspose2d(1, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.transposed_conv2 = nn.ConvTranspose2d(16, 8, 5)
        self.bn2 = nn.BatchNorm2d(8)
        self.transposed_conv3 = nn.ConvTranspose2d(8, 3, 5)

    def forward_2(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = x.view(-1, 1, 52, 52)
        x = F.leaky_relu(self.bn1(self.transposed_conv1(x)))
        x = F.leaky_relu(self.bn2(self.transposed_conv2(x)))
        x = torch.tanh(self.transposed_conv3(x))
        return x

# ============== Celebs: Generator Model_3 ============== #

    def set_model_3(self):
        self.tconv1 = nn.ConvTranspose2d(self.input_size, self.d*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d*8)
        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.d*8, self.d*4, 4, stride=2,  padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d*4)
        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.d*4, self.d*2, 4, stride=2,  padding = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d*2)
        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.d*2, self.d, 4, stride=2,  padding = 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d)
        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.d, 3, 4, stride=2,  padding = 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward_3(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))# do not use sigmoid! only tanh
        return x

# ============== Celebs: Generator Model_4 ============== #

    def set_model_4(self):
        self.tconv1 = nn.ConvTranspose2d(self.input_size, self.d*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d*8)
        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.d*8, self.d*4, 4, stride=2,  padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d*4)
        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.d*4, self.d*2, 4, stride=2,  padding = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d*2)
        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.d*2, self.d, 4, stride=2,  padding = 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d)
        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.d, 1, 4, stride=2,  padding = 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward_4(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = x = F.leaky_relu(self.bn1(self.tconv1(x)),0.2)
        x = x = F.leaky_relu(self.bn2(self.tconv2(x)),0.2)
        x = x = F.leaky_relu(self.bn3(self.tconv3(x)),0.2)
        x = x = F.leaky_relu(self.bn4(self.tconv4(x)),0.2)

        x = torch.tanh(self.tconv5(x))# do not use sigmoid! only tanh
        return x

# ============== Celebs: Generator Model_5 ============== #

    def set_model_5(self):
        self.tconv1 = nn.ConvTranspose2d(self.input_size, self.d*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d*8)
        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.d*8, self.d*4, 4, stride=2,  padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d*4)
        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.d*4, self.d*2, 4, stride=2,  padding = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d*2)
        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.d*2, self.d, 4, stride=2,  padding = 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d)
        # Input Dimension: (ngf) * 32 * 32
        self.tconv5 = nn.ConvTranspose2d(self.d, 3, 4, stride=2,  padding = 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    def forward_5(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = F.dropout(x, 0.2)
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv5(x))# do not use sigmoid! only tanh
        return x

# ============== Celebs: Generator Model_6 ============== #

    '''NOTE: This is for downscaled data!!'''

    #  NOTE: This is for downscaled data!!
    def set_model_6(self):
        self.d = int(self.d/2)
        self.tconv1 = nn.ConvTranspose2d(self.input_size, self.d*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d*8)
        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.d*8, self.d*4, 4, stride=2,  padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d*4)
        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.d*4, self.d*2, 4, stride=2,  padding = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d*2)
        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.d*2, 3, 4, stride=2,  padding = 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(self.d)
        # Input Dimension: (ngf) * 32 * 32
        # self.tconv5 = nn.ConvTranspose2d(self.d, 3, 4, stride=2,  padding = 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    #  NOTE: This is for downscaled data!!
    def forward_6(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = F.dropout(x, 0.2)
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        # x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv4(x))# do not use sigmoid! only tanh
        return x

# ============== Celebs: Generator Model_7 ============== #

    '''same as model 5 but with d=d/2 '''

    def set_model_7(self):
        self.d = int(self.d/2)
        self.set_model_5()

    def forward_7(self, input):
        return self.forward_5(input)

# ============== Celebs: Generator Model_8 ============== #

    '''same as model 5 but with d=d*2 '''

    def set_model_8(self):
        self.d = int(self.d*2)
        self.set_model_5()

    def forward_8(self, input):
        return self.forward_5(input)

# ============== Celebs: Generator Model_9 ============== #

    '''NOTE: This is for downscaled data!! 
        same as model 6 but without d=d/2 '''
    
    #  NOTE: This is for downscaled data!!
    def set_model_9(self):
        self.tconv1 = nn.ConvTranspose2d(self.input_size, self.d*8, kernel_size=4, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d*8)
        # Input Dimension: (ngf*8) x 4 x 4
        self.tconv2 = nn.ConvTranspose2d(self.d*8, self.d*4, 4, stride=2,  padding = 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d*4)
        # Input Dimension: (ngf*4) x 8 x 8
        self.tconv3 = nn.ConvTranspose2d(self.d*4, self.d*2, 4, stride=2,  padding = 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d*2)
        # Input Dimension: (ngf*2) x 16 x 16
        self.tconv4 = nn.ConvTranspose2d(self.d*2, 3, 4, stride=2,  padding = 1, bias=False)
        # self.bn4 = nn.BatchNorm2d(self.d)
        # Input Dimension: (ngf) * 32 * 32
        # self.tconv5 = nn.ConvTranspose2d(self.d, 3, 4, stride=2,  padding = 1, bias=False)
        #Output Dimension: (nc) x 64 x 64

    #  NOTE: This is for downscaled data!!
    def forward_9(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = F.dropout(x, 0.2)
        x = F.relu(self.bn1(self.tconv1(x)))
        x = F.relu(self.bn2(self.tconv2(x)))
        x = F.relu(self.bn3(self.tconv3(x)))
        # x = F.relu(self.bn4(self.tconv4(x)))

        x = torch.tanh(self.tconv4(x))# do not use sigmoid! only tanh
        return x

# ============== Celebs: Generator Model_10 ============== #

    '''same as model 1 but with d=d*2 '''

    def set_model_10(self):
        self.d = int(self.d*2)
        self.set_model_1()

    def forward_10(self, input):
        return self.forward_1(input)

# ============== Celebs: Generator Model_11 ============== #

    '''From: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''

    def set_model_11(self):
        self.tconv1 = nn.ConvTranspose2d( self.input_size, self.d * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d * 8)
        self.tconv2 = nn.ConvTranspose2d(self.d * 8, self.d * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d * 4)
        self.tconv3 = nn.ConvTranspose2d( self.d * 4, self.d * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d * 2)
        self.tconv4 = nn.ConvTranspose2d( self.d * 2, self.d, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d)
        self.tconv5 = nn.ConvTranspose2d( self.d, 3, 4, 2, 1, bias=False)

    def forward_11(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = F.relu(self.bn1(self.tconv1(x)), True)
        x = F.relu(self.bn2(self.tconv2(x)), True)
        x = F.relu(self.bn3(self.tconv3(x)), True)
        x = F.relu(self.bn4(self.tconv4(x)), True)
        x = torch.tanh(self.tconv5(x))
        return x

# ============== Celebs: Generator Model_12 ============== #

    '''same as model 1 but with d=d*4 '''

    def set_model_12(self):
        self.d = int(self.d*4)
        self.set_model_1()

    def forward_12(self, input):
        return self.forward_1(input)

# ============== Celebs: Generator Model_13 ============== #

    '''same as model 11 but with d=d*2 '''

    def set_model_13(self):
        self.d = int(self.d*2)
        self.set_model_11()

    def forward_13(self, input):
        return self.forward_11(input)

# ============== Celebs: Generator Model_14 ============== #

    ''' Based on 11 but with drops '''

    def set_model_14(self):
        self.tconv1 = nn.ConvTranspose2d( self.input_size, self.d * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d * 8)
        self.tconv2 = nn.ConvTranspose2d(self.d * 8, self.d * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d * 4)
        self.tconv3 = nn.ConvTranspose2d( self.d * 4, self.d * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d * 2)
        self.tconv4 = nn.ConvTranspose2d( self.d * 2, self.d, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d)
        self.tconv5 = nn.ConvTranspose2d( self.d, 3, 4, 2, 1, bias=False)

    def forward_14(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = F.relu(self.bn1(self.tconv1(x)), True)
        x = F.relu(self.bn2(self.tconv2(x)), True)
        x = F.relu(self.bn3(self.tconv3(x)), True)
        x = F.dropout(x, 0.1)
        x = F.relu(self.bn4(self.tconv4(x)), True)
        x = F.dropout(x, 0.1)
        x = torch.tanh(self.tconv5(x))
        return x

# ============== Celebs: Generator Model_15 ============== #

    '''From: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''

    def set_model_15(self):
        self.tconv1 = nn.ConvTranspose2d( self.input_size, self.d * 8, 4, 1, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.d * 8)
        self.tconv2 = nn.ConvTranspose2d(self.d * 8, self.d * 4, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d * 4)
        self.tconv3 = nn.ConvTranspose2d( self.d * 4, self.d * 2, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d * 2)
        self.tconv4 = nn.ConvTranspose2d( self.d * 2, self.d, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d)
        self.tconv5 = nn.ConvTranspose2d( self.d, 3, 4, 2, 1, bias=False)

    def forward_15(self, input):
        x = input.view(-1, self.input_size, 1, 1)
        x = F.relu(self.bn1(self.tconv1(x)), True)
        x = F.relu(self.bn2(self.tconv2(x)), True)
        x = F.relu(self.bn3(self.tconv3(x)), True)
        x = F.relu(self.bn4(self.tconv4(x)), True)
        x = F.dropout(x, 0.1)
        x = torch.tanh(self.tconv5(x))
        return x
        
# =============================================== #

class Discriminator_celeb(BaseModel):
    # initializers
    def __init__(self, model_name, d):
        super(Discriminator_celeb, self).__init__("Discriminator_celeb", model_name)
        self.d = d
        self.set_model()

# ============== Celebs: Discriminator Model_1 ============== #

    def set_model_1(self):
        self.conv1 = nn.Conv2d(3, self.d, 4, stride=2)
        self.conv2 = nn.Conv2d(self.d, self.d*2, 4, stride=2)
        self.bn1 = nn.BatchNorm2d(self.d*2)
        self.conv3 = nn.Conv2d(self.d*2, self.d*4, 4, stride=2)
        self.bn2 = nn.BatchNorm2d(self.d*4)
        self.conv4 = nn.Conv2d(self.d*4, self.d*8, 4, stride=2)
        self.bn3 = nn.BatchNorm2d(self.d*8)
        self.fc1 = nn.Linear(self.d*8*2*2, 1, True)

    def forward_1(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = x.view(-1, self.d * 8 * 2 * 2)
        # x = torch.tanh(self.fc1(x)) # do not use sigmoid! only tanh
        x = torch.sigmoid(self.fc1(x)) # do not use sigmoid! only tanh
        return x

# ============== Celebs: Discriminator Model_2 ============== #

    def set_model_2(self):
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        # self.conv4 = nn.Conv2d(self.d*4, self.d*8, 4, stride=2)
        # self.bn3 = nn.BatchNorm2d(self.d*8)
        self.fc1 = nn.Linear(64*9*9, 64*4*4)
        self.fc2 = nn.Linear(64*4*4, 64)
        self.fc3 = nn.Linear(64, 1)
        # self.fc2 = nn.Linear(, 512)

    def forward_2(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = self.pool(x)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = self.pool(x)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.dropout(x, 0.3)
        x = x.view(-1,64*9*9)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.tanh(self.fc3(x)) # do not use sigmoid! only tanh
        return x

# ============== Celebs: Discriminator Model_3 ============== #

    def set_model_3(self):
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 64, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 256, 5)
        self.bn2 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, 3)
        self.bn3 = nn.BatchNorm2d(512)
        self.fc1 = nn.Linear(512*2*2, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 32)
        self.fc4 = nn.Linear(32, 1)

    def forward_3(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = x.view(-1,512*2*2)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        x = torch.tanh(self.fc4(x)) # do not use sigmoid! only tanh
        return x

# ============== Celebs: Discriminator Model_4 ============== #

    def set_model_4(self):
        self.conv1 = nn.Conv2d(3, 16, 5)
        self.conv2 = nn.Conv2d(16, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 128, 5)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128*5*5, 1024)
        # self.fc1 = nn.Linear(512*2*2, 512)
        self.fc2 = nn.Linear(1024, 64)
        # self.fc3 = nn.Linear(512, 64)
        self.fc4 = nn.Linear(64, 1)

    def forward_4(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.bn1(self.conv2(x)), 0.2)
        x = self.pool(x)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        # x = self.pool(x)
        x = F.leaky_relu(self.bn3(self.conv4(x)), 0.2)
        x = x.view(-1,128*5*5)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        # x = F.leaky_relu(self.fc3(x), 0.2)
        # x = F.dropout(x, 0.3)
        x = torch.tanh(self.fc4(x)) # do not use sigmoid! only tanh
        return x

# ============== Celebs: Discriminator Model_5 ============== #

    def set_model_5(self):
        self.cv1 = nn.Conv2d(3, self.d, kernel_size=4, stride=2, padding=1, bias=False) # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(self.d, self.d*2, 4, stride=2, padding=1 ) # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(self.d*2) # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(self.d*2, self.d*4, 4, stride=2, padding=1) # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(self.d*4)
        self.cv4 = nn.Conv2d(self.d*4, self.d*8, 4, stride=2, padding=1, bias=False) # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(self.d* 8)
        self.cv5 = nn.Conv2d(self.d*8, 1, 4, stride=1, padding=0, bias=False) # (512, 4, 4) -> (1, 1, 1)

    def forward_5(self, input):
        x = F.leaky_relu(self.cv1(input))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        # x = torch.sigmoid(self.cv5(x)) # don't use tanh?
        x = torch.tanh(self.cv5(x)) # don't use tanh?
        return x.view(-1, 1)

# ============== Celebs: Discriminator Model_6 ============== #

    def set_model_6(self):
        self.cv1 = nn.Conv2d(1, self.d, kernel_size=4, stride=2, padding=1, bias=False) # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(self.d, self.d*2, 4, stride=2, padding=1 ) # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(self.d*2) # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(self.d*2, self.d*4, 4, stride=2, padding=1) # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(self.d*4)
        self.cv4 = nn.Conv2d(self.d*4, self.d*8, 4, stride=2, padding=1, bias=False) # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(self.d* 8)
        self.cv5 = nn.Conv2d(self.d*8, 1, 4, stride=1, padding=0, bias=False) # (512, 4, 4) -> (1, 1, 1)

    def forward_6(self, input):
        x = torch.mean(input,1)
        x = x.view(-1, 1, x.shape[1], x.shape[2])
        x = F.leaky_relu(self.cv1(x))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        x = torch.sigmoid(self.cv5(x)) # don't use tanh?
        return x.view(-1, 1)

# ============== Celebs: Discriminator Model_7 ============== #

    '''same as model 6 with a few dropouts and tanh instead of sigmoid'''

    def set_model_7(self):
        self.cv1 = nn.Conv2d(3, self.d, kernel_size=4, stride=2, padding=1, bias=False) # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(self.d, self.d*2, 4, stride=2, padding=1 ) # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(self.d*2) # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(self.d*2, self.d*4, 4, stride=2, padding=1) # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(self.d*4)
        self.cv4 = nn.Conv2d(self.d*4, self.d*8, 4, stride=2, padding=1, bias=False) # (256, 8, 8) -> (512, 4, 4)
        self.bn4 = nn.BatchNorm2d(self.d* 8)
        self.cv5 = nn.Conv2d(self.d*8, 1, 4, stride=1, padding=0, bias=False) # (512, 4, 4) -> (1, 1, 1)

    def forward_7(self, input):
        dropout = 0.2
        x = F.leaky_relu(self.cv1(input))
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.dropout(x, dropout)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.dropout(x, dropout)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        x = torch.tanh(self.cv5(x)) # don't use tanh?
        return x.view(-1, 1)

# ============== Celebs: Discriminator Model_8 ============== #

    '''NOTE: This is for downscaled data!!'''

    #  NOTE: This is for downscaled data!!
    def set_model_8(self):
        self.cv1 = nn.Conv2d(3, self.d, kernel_size=4, stride=2, padding=1, bias=False) # (3, 64, 64) -> (64, 32, 32)
        self.cv2 = nn.Conv2d(self.d, self.d*2, 4, stride=2, padding=1 ) # (64, 32, 32) -> (128, 16, 16)
        self.bn2 = nn.BatchNorm2d(self.d*2) # spatial batch norm is applied on num of channels
        self.cv3 = nn.Conv2d(self.d*2, self.d*4, 4, stride=2, padding=1) # (128, 16, 16) -> (256, 8, 8)
        self.bn3 = nn.BatchNorm2d(self.d*4)
        self.cv4 = nn.Conv2d(self.d*4, 1, 4, stride=2, padding=0, bias=False) # (256, 8, 8) -> (512, 4, 4)
        # self.bn4 = nn.BatchNorm2d(self.d* 8)
        # self.cv5 = nn.Conv2d(self.d*8, 1, 4, stride=1, padding=0, bias=False) # (512, 4, 4) -> (1, 1, 1)

    #  NOTE: This is for downscaled data!!
    def forward_8(self, input):
        dropout = 0.2
        x = F.leaky_relu(self.cv1(input))
        # x = F.dropout(x, dropout)
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, True)
        x = F.dropout(x, dropout)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, True)
        x = F.dropout(x, dropout)
        # x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, True)
        # x = F.dropout(x, dropout)
        # x = torch.sigmoid(self.cv5(x)) # don't use tanh?
        x = torch.tanh(self.cv4(x)) # don't use tanh?
        return x.view(-1, 1)

# ============== Celebs: Discriminator Model_9 ============== #

    ''' same as model 7 but without d=d/2 '''

    def set_model_9(self):
        self.d = int(self.d/2)
        self.set_model_7()

    def forward_9(self, input):
        return self.forward_7(input)

# ============== Celebs: Discriminator Model_10 ============== #

    ''' same as model 1 but without d=d*4 '''

    def set_model_10(self):
        self.d = int(self.d*4)
        self.set_model_1()

    def forward_10(self, input):
        return self.forward_1(input)

# ============== Celebs: Discriminator Model_11 ============== #

    ''' same as model 1 but without d=d*2 '''
    def set_model_11(self):
        self.d = int(self.d*2)
        self.set_model_1()

    def forward_11(self, input):
        return self.forward_1(input)

# ============== Celebs: Discriminator Model_12 ============== #

    '''From: https://pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html'''
    def set_model_12(self):
        self.cv1 = nn.Conv2d(3, self.d, 4, 2, 1, bias=False)
        self.cv2 = nn.Conv2d(self.d, self.d * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d * 2)
        self.cv3 = nn.Conv2d(self.d*2, self.d * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d * 4)
        self.cv4 = nn.Conv2d(self.d*4, self.d * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d * 8)
        self.cv5 = nn.Conv2d(self.d*8, 1, 4, 2, 0, bias=False)

    def forward_12(self, input):
        x = F.leaky_relu(self.cv1(input), 0.2, inplace=True)
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, inplace=True)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, inplace=True)
        x = F.dropout(x, 0.35)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, inplace=True)
        x = F.dropout(x, 0.35)
        x = torch.sigmoid(self.cv5(x))
        return x.view(-1, 1)

# ============== Celebs: Discriminator Model_13 ============== #

    ''' same as model 12 but without d=d*2 '''
    def set_model_13(self):
        self.d = int(self.d*2)
        self.set_model_12()

    def forward_13(self, input):
        return self.forward_12(input)

# ============== Celebs: Discriminator Model_14 ============== #

    '''Based on 12 with less drops'''
    def set_model_14(self):
        self.cv1 = nn.Conv2d(3, self.d, 4, 2, 1, bias=False)
        self.cv2 = nn.Conv2d(self.d, self.d * 2, 4, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(self.d * 2)
        self.cv3 = nn.Conv2d(self.d*2, self.d * 4, 4, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.d * 4)
        self.cv4 = nn.Conv2d(self.d*4, self.d * 8, 4, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(self.d * 8)
        self.cv5 = nn.Conv2d(self.d*8, 1, 4, 2, 0, bias=False)

    def forward_14(self, input):
        x = F.leaky_relu(self.cv1(input), 0.2, inplace=True)
        x = F.dropout(x, 0.1)
        x = F.leaky_relu(self.bn2(self.cv2(x)), 0.2, inplace=True)
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.bn3(self.cv3(x)), 0.2, inplace=True)
        x = F.dropout(x, 0.25)
        x = F.leaky_relu(self.bn4(self.cv4(x)), 0.2, inplace=True)
        x = F.dropout(x, 0.25)
        x = torch.sigmoid(self.cv5(x))
        return x.view(-1, 1)

# ======================================================== #
# ========================= MNIST ======================== #
# ======================================================== #

class Generator_mnist(BaseModel):
    # initializers
    def __init__(self, model_name, input_size=100, n_class=28*28):
        super(Generator_mnist, self).__init__("Generator_mnist", model_name)
        self.n_class = n_class
        self.input_size = input_size
        self.set_model()

# ============== MNIST: Generator Model_1 ============== #

    def set_model_1(self):
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.transposed_conv1 = nn.ConvTranspose2d(4, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.transposed_conv2 = nn.ConvTranspose2d(16, 8, 5)
        self.bn2 = nn.BatchNorm2d(8)
        self.transposed_conv3 = nn.ConvTranspose2d(8, 1, 5)

    def forward_1(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = x.view(-1,4,16, 16)
        x = F.leaky_relu(self.bn1(self.transposed_conv1(x)))
        x = F.leaky_relu(self.bn2(self.transposed_conv2(x)))
        x = torch.tanh(self.transposed_conv3(x))
        return x

# ============== MNIST: Generator Model_2 ============== #

    def set_model_2(self):
        #no fully connected layers
        self.transposed_conv1 = nn.ConvTranspose2d(4, 48, 4, stride=2)
        self.bn1 = nn.BatchNorm2d(48)
        self.transposed_conv2 = nn.ConvTranspose2d(48, 24, 5)
        self.bn2 = nn.BatchNorm2d(24)
        self.transposed_conv3 = nn.ConvTranspose2d(24, 12, 5)
        self.bn3 = nn.BatchNorm2d(12)
        self.transposed_conv4 = nn.ConvTranspose2d(12, 6, 5)
        self.bn4 = nn.BatchNorm2d(6)
        self.transposed_conv5 = nn.ConvTranspose2d(6, 1, 5)

    def forward_2(self, input):
        x = input.view(-1, 4, 5, 5)
        x = F.leaky_relu(self.bn1(self.transposed_conv1((x))), 0.2)
        x = F.leaky_relu(self.bn2(self.transposed_conv2((x))), 0.2)
        x = F.leaky_relu(self.bn3(self.transposed_conv3((x))), 0.2)
        x = F.leaky_relu(self.bn4(self.transposed_conv4((x))), 0.2)
        # x = torch.sigmoid(self.transposed_conv5((x)))
        x = torch.tanh(self.transposed_conv5((x)))
        return x

# ============== MNIST: Generator Model_3 ============== #

    def set_model_3(self):
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.fc4 = nn.Linear(self.fc3.out_features, self.n_class)

    def forward_3(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = torch.tanh(self.fc4(x))
        # x = torch.sigmoid(self.fc4(x))
        x = x.view(-1, 1, 28, 28)
        return x

# ============== MNIST: Generator Model_4 ============== #
    
    '''based on model_1 with added dropouts'''

    def set_model_4(self):
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.transposed_conv1 = nn.ConvTranspose2d(4, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.transposed_conv2 = nn.ConvTranspose2d(16, 8, 5)
        self.bn2 = nn.BatchNorm2d(8)
        self.transposed_conv3 = nn.ConvTranspose2d(8, 1, 5)

    def forward_4(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = x.view(-1,4,16, 16)
        # x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.bn1(self.transposed_conv1(x)))
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.bn2(self.transposed_conv2(x)))
        x = F.dropout(x, 0.2)
        x = torch.tanh(self.transposed_conv3(x))
        return x

# ============== MNIST: Generator Model_5 ============== #
    
    '''based on model_4 with dropouts between the conv layers instead of the fully connected'''

    def set_model_5(self):
        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 1024)
        self.transposed_conv1 = nn.ConvTranspose2d(4, 16, 5)
        self.bn1 = nn.BatchNorm2d(16)
        self.transposed_conv2 = nn.ConvTranspose2d(16, 8, 5)
        self.bn2 = nn.BatchNorm2d(8)
        self.transposed_conv3 = nn.ConvTranspose2d(8, 1, 5)

    def forward_5(self, input):
        x = F.leaky_relu(self.fc1(input), 0.2)
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = x.view(-1,4,16, 16)
        x = F.leaky_relu(self.bn1(self.transposed_conv1(x)))
        x = F.leaky_relu(self.bn2(self.transposed_conv2(x)))
        x = torch.tanh(self.transposed_conv3(x))
        return x

# =============================================== #

class Discriminator_mnist(BaseModel):
    
    def __init__(self, model_name, input_size=28*28, n_class=1):
        super(Discriminator_mnist, self).__init__("Discriminator_mnist", model_name)
        self.n_class = n_class
        self.input_size = input_size
        self.set_model()

# ============== MNIST: Discriminator Model_1 ============== #

    def set_model_1(self):
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24*20*20, 64)
        self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, self.n_class)

    def forward_1(self, input):
        x = F.leaky_relu(self.conv1(input))
        x = F.leaky_relu(self.conv2(x))
        x = x.view(-1, 24 * 20 * 20)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.2)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.1)
        x = torch.tanh(self.fc4(x))
        # x = torch.sigmoid(self.fc4(x))
        return x

# ============== MNIST: Discriminator Model_2 ============== #

    def set_model_2(self):
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.bn1 = nn.BatchNorm2d(24)
        self.conv3 = nn.Conv2d(24, 48, 5)
        self.bn2 = nn.BatchNorm2d(48)
        self.conv4 = nn.Conv2d(48, self.n_class, 4)


    def forward_2(self, input):
        x = F.leaky_relu(self.conv1(input), 0.2)
        x = F.leaky_relu(self.pool(self.bn1(self.conv2(x))), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.bn2(self.conv3(x)), 0.2)
        x = F.dropout(x, 0.2)
        x = torch.tanh(self.pool(self.conv4(x)))
        # x = torch.sigmoid(self.pool(self.conv4(x)))
        x = x.view(-1, 1)
        return x

# ============== MNIST: Discriminator Model_3 ============== #

    def set_model_3(self):
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.n_class)

    def forward_3(self, input):
        x = input.view(-1, input.shape[2]*input.shape[3])
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.tanh(self.fc4(x)) # NOTE sigmoid is terrible. only use tanh

# ============== MNIST: Discriminator Model_4 ============== #

    def set_model_4(self):
        self.fc1 = nn.Linear(self.input_size, 1024)
        self.fc2 = nn.Linear(self.fc1.out_features, self.fc1.out_features // 2)
        self.fc3 = nn.Linear(self.fc2.out_features, self.fc2.out_features // 2)
        self.fc4 = nn.Linear(self.fc3.out_features, self.n_class)

    def forward_4(self, input):
        x = input.view(-1, input.shape[2]*input.shape[3])
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        return torch.sigmoid(self.fc4(x))

# ============== MNIST: Discriminator Model_5 ============== #

    def set_model_5(self):
        self.conv1 = nn.Conv2d(1, 12, 5)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.pool = nn.MaxPool2d(2, 2)
        # self.fc1 = nn.Linear(24*12*12, 64)
        # self.fc2 = nn.Linear(self.fc1.out_features, 512)
        self.fc3 = nn.Linear(24*4*4, 256)
        # self.fc3 = nn.Linear(self.fc2.out_features, 256)
        self.fc4 = nn.Linear(self.fc3.out_features, self.n_class)

    def forward_5(self, input):
        x = F.leaky_relu(self.conv1(input))
        x = self.pool(x)
        x = F.leaky_relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(-1, 24 * 4 * 4)
        # x = x.view(-1, 24 * 20 * 20)
        # x = F.leaky_relu(self.fc1(x), 0.2)
        # x = F.dropout(x, 0.3)
        # x = F.leaky_relu(self.fc2(x), 0.2)
        x = F.dropout(x, 0.3)
        x = F.leaky_relu(self.fc3(x), 0.2)
        x = F.dropout(x, 0.3)
        # x = F.dropout(x, 0.1)
        x = torch.tanh(self.fc4(x))
        # x = torch.sigmoid(self.fc4(x))
        return x

# =================================================== #