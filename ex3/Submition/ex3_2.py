import os
import torch
import socket
import argparse
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from itertools import combinations

from models import AE
from common import get_dataset, read_args_from_file, get_data_lists_by_label
from utils import SavingPath, init_randomness, init_flags

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # needed in case of conflict error with numpy

# ◄►◄►◄►◄►◄► Magic Variables ◄►◄►◄►◄►◄► #

PATH_Q1 = "em-M1_dm-M1_e-50_b-48_z-10_lr-0.001_name-urim_auto_encoder_param-79.pkl"
PATH_Q4 = "em-M1_dm-M1_e-100_b-48_z-10_lr-0.0005_name-urim_auto_encoder_param_method-5.pkl"

PATH_TO_MAN_IMAGE = "man.png"

DEFAULT_PATH_Q = "Q4"

NUM_OF_IMAGES = 300
# NUM_OF_IMAGES = 2000

# DEBUG_MODE = True
DEBUG_MODE = False

# CPU_MODE = True
CPU_MODE = False

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#

def view_latent_space_by_class(ae, z_size, present_outlier = True):
  '''creates scattering plots of the z latent space and saves them to file'''

  ae.eval()
  dataset_labels_dict = get_data_lists_by_label() # get a split dict of the dataset by label

  # gets all the pairs of indices for the scattering. this is just n choose k
  combs = list(combinations(list(range(0,z_size)), 2)) 
  combs_len = len(combs)

  # iterating over all combinations and saving them
  for i,comb in enumerate(combs):
    for digit in range(0, 10):
      print(f"{i}/{combs_len} - {comb}:{digit}") # give feedback
      images = dataset_labels_dict[digit][:NUM_OF_IMAGES] #slice the amount of images we want to plot
      # get the represanting z vectors from the images
      z_vectors = np.array(list(map(lambda x: ae.encoder(x).cpu().detach().numpy(), images))).reshape((NUM_OF_IMAGES, ae.decoder.z_size))
      plt.scatter(z_vectors[:,comb[0]], z_vectors[:,comb[1]], alpha=0.4, s=10)

    if present_outlier: # in case we also want to plot the out-of-class images
      base_image = dataset_labels_dict[0][0] # get some image as a baseline images for the outlier. This is a random
      
      # outlier_1 - const image
      outlier = base_image*0+0.5
      z_out = ae.encoder(outlier).cpu().detach().numpy()
      plt.scatter(z_out[:,comb[0]], z_out[:,comb[1]], alpha=1, s=25, c='#FFFF00', edgecolors='black', label="const")
      
      # outlier_2 - const + noise image
      outlier = torch.clip(base_image*0+0.5 + (torch.randn(outlier.size())-0.5) * 0.5,-1,1)
      z_out = ae.encoder(outlier).cpu().detach().numpy()
      plt.scatter(z_out[:,comb[0]], z_out[:,comb[1]], alpha=1, s=25, c='#00FF00', edgecolors='black', label="const+noise")
      
      # outlier_3 - inverse image
      outlier = base_image*(-1) # to get the inverse we only need to multiply by -1 since the image is in [-1,1]!!
      z_out = ae.encoder(outlier).cpu().detach().numpy()
      plt.scatter(z_out[:,comb[0]], z_out[:,comb[1]], alpha=1, s=25, c='#FF0000', edgecolors='black', label="negative")
      
      # outlier_4 - the original image flipped on the horizontal axis
      outlier = torch.flip(base_image,[3])
      z_out = ae.encoder(outlier).cpu().detach().numpy()
      plt.scatter(z_out[:,comb[0]], z_out[:,comb[1]], alpha=1, s=25, c='#00BFFF', edgecolors='black', label="flipped")

      # outlier_5 - the man sketch
      man = Image.open(PATH_TO_MAN_IMAGE)
      man = (np.asarray(man)/255 - 0.5)/0.5 # this is the transform applied to MNIST by torch
      outlier = base_image
      outlier[0,0,:,:] = torch.as_tensor(man)
      z_out = ae.encoder(outlier).cpu().detach().numpy()
      plt.scatter(z_out[:,comb[0]], z_out[:,comb[1]], alpha=1, s=25, c='Black', edgecolors='black', label="man")

    # always set the limits to [0,1] for easier comparison of the scattering plots
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend() # only for the out-of-class scattering
    plt.savefig(SavingPath.get_path(f"_combined_dim1-{comb[0]}_dim2-{comb[1]}.png"), dpi = 100)
    plt.clf()

def main():

  torch.autograd.set_detect_anomaly(True) # set to true to help debug
  
  # get CLI args
  parser = argparse.ArgumentParser(description='Ex3_2')
  
  parser.add_argument('-q', type=str, default=DEFAULT_PATH_Q, metavar=('question path'),
                        help=f'Sets if to train the Q1 or Q4 model (default: {DEFAULT_PATH_Q})')
  parser.add_argument('--str', type=str, default="", metavar=('str'),
                        help=f'Appends a string to the end of the path name. (default: "")')
  parser.add_argument('-o','--outlier', action='store_true', default=True,
                      help=f'Adds outliers to the scattering (default: True)')
  args = parser.parse_args()

  # initialization
  init_randomness()
  init_flags(CPU_MODE, DEBUG_MODE)
  
  # set parameters according to input args
  if(args.q.lower() == "q1"):
    path = PATH_Q1
  elif(args.q.lower() == "q4"):
    path = PATH_Q4
  else: raise Exception(f"No such Q path: {args.q}")

  file_args = read_args_from_file(path)
  if(DEBUG_MODE):
      SavingPath(f"Debug_name-{socket.gethostname()}_qPath-{args.q.upper()}")
  else:
    SavingPath(f'scatter_em-{file_args["e_model"]}_dm-{file_args["d_model"]}_b-{file_args["batch"]}_z-{file_args["z"]}_name-{socket.gethostname()}_qPath-{args.q.upper()}')

  if(args.str != ""): SavingPath.append(f"_{args.str}")

  dataset, _ = get_dataset(file_args["batch"])
  ae = AE(file_args["e_model"], file_args["d_model"], file_args["z"])
  ae.load_state_dict(torch.load(path, map_location=torch.device('cpu')))

  # create the scatterings
  view_latent_space_by_class(ae, 10, args.outlier)

  # wrap up
  SavingPath.finish("-Done")
  print(f"Current dir path:\n{SavingPath.get_dir_path()}")

if __name__ == '__main__':
  # change the current directory to the location of the file to read the model and save images and plots in the right location
  os.chdir(os.path.dirname(os.path.realpath(__file__)))
  main()