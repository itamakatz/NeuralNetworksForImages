import torch
import torch.nn as nn
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import Dataset

from models import Discriminator_celeb, Generator_celeb, Generator_mnist, Discriminator_mnist, \
    NonSaturatedCrossEntropy, CrossEntropy, AE, BCEWithLogitsLoss

# ◄►◄► Configuration Parser ◄►◄► #
class Celeb64(Dataset):
    def __init__(self, file_name, transform=None):
        self.data = torch.from_numpy(np.transpose(np.load(file_name),(0,3,1,2)).astype(np.int8))
        self.transform = transform

    def __getitem__(self, index):
        return self.transform(self.data[index].type(torch.uint8))

    def __len__(self):
        return len(self.data)

def get_loss(loss_name):

    if("entropy" in loss_name.lower() or loss_name == "H" or loss_name == "CE"): loss_name = "cross_entropy"

    loss_dict = {
        "mse".lower(): (nn.MSELoss(), nn.MSELoss()),
        "non_saturated".lower(): (NonSaturatedCrossEntropy(), BCEWithLogitsLoss()),
        "cross_entropy".lower(): (CrossEntropy(), BCEWithLogitsLoss()),
    }
    if(loss_name.lower() in loss_dict): return loss_dict[loss_name.lower()]
    else: raise Exception(f"No such loss function named: {loss_name.lower()}")


def get_dataset(dataset_name, batch_size):
    if(dataset_name.lower() == "MNIST".lower()):
        transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
        dataset = torch.utils.data.DataLoader(datasets.MNIST('data', train=True, download=True, transform=transform_mnist), batch_size=batch_size, shuffle=True)
        return (dataset, (28,28))
    elif(dataset_name.lower() == "celeb".lower()):
        transform_celeb = transforms.Compose([transforms.ToPILImage(), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
        celeb64 = Celeb64("./data/CelebA64.npy", transform_celeb)
        dataset = torch.utils.data.DataLoader(celeb64,  batch_size=batch_size, shuffle=True)
        return (dataset, (64,64))
    else: raise Exception(f"No such dataset named: {dataset_name.lower()}")

def get_models(args, im_size, CPU_MODE):
    if(args.dataset.lower() == "MNIST".lower()):
        gen = Generator_mnist(args.gen_model, input_size=args.z_size, n_class=im_size[0]*im_size[1])
        dis = Discriminator_mnist(args.dis_model, input_size=im_size[0]*im_size[1], n_class=1)
    elif(args.dataset.lower() == "celeb".lower()):
        gen = Generator_celeb(args.gen_model, args.z_size, 64)
        dis = Discriminator_celeb(args.dis_model, 64)
        gen.apply(weights_init)
        dis.apply(weights_init)
    else: raise Exception(f"No such dataset named: {args.dataset.lower()}")
    if(torch.cuda.is_available() and not CPU_MODE):
        gen, dis = gen.cuda(), dis.cuda()
    return (gen,dis)

def get_auto_encoder(args, CPU_MODE):
    ae = AE(args.dis_model, args.gen_model, args.z_size)
    if(torch.cuda.is_available() and not CPU_MODE):
        ae = ae.cuda()
    return ae

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Conv') != -1:
#         nn.init.normal_(m.weight.data, 0.0, 0.02)
#     elif classname.find('BatchNorm') != -1:
#         nn.init.normal_(m.weight.data, 1.0, 0.02)
#         nn.init.constant_(m.bias.data, 0)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
