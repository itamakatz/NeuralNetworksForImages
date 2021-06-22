import os
import torch
import random
from torchvision import datasets, transforms

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

def get_dataset(batch):
  '''get the MNIST data set'''
  transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
  dataset = torch.utils.data.DataLoader(datasets.MNIST('./data', train=True, download=True, transform=transform_mnist), batch_size=batch, shuffle=True)
  return (dataset, (28,28))

def get_data_as_list(dataset, count = 1):
  '''get a number of images from the data set in a list'''
  datalist = [torch.unsqueeze(next(iter(dataset))[0][0], 0) for i in range(count*10)]
  random.shuffle(datalist)
  return datalist[:count]

def get_data_lists_by_label():
  '''returns a dict with lists split by label'''
  transform_mnist = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
  dataset = torch.utils.data.DataLoader(
    datasets.MNIST('data', train=True, download=True, transform=transform_mnist),
    batch_size=1, shuffle=True)

  dataset_dict = {}
  for item in dataset:
    image, label = item
    if(dataset_dict.get(label.item()) == None):
      dataset_dict[label.item()] = [image]
    else:
      dataset_dict[label.item()].append(image)

  return dataset_dict

def read_args_from_file(path):
  '''returns the info of a model according to its name'''
  z = [int(s) for s in (path.split("z-")[1].split("_")) if s.isdigit()][0]
  batch_size = [int(s) for s in (path.split("b-")[1].split("_")) if s.isdigit()][0]
  e_model = [s for s in (path.split("em-")[1].split("_"))][0]
  d_model = [s for s in (path.split("dm-")[1].split("_"))][0]

  return {"z": z, "batch": batch_size, "e_model": e_model, "d_model": d_model, "path": path}