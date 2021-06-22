import os
import torch
import socket
import random
import argparse
import numpy as np
from PIL import Image
import torch.nn as nn
from datetime import datetime

from models import AE
from common import get_dataset, get_data_as_list, read_args_from_file
from utils import SavingPath, init_randomness, init_flags, Progress, save_tensor_im, save_loss_plot

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # needed in case of conflict error with numpy

# ◄►◄►◄►◄►◄► Magic Variables ◄►◄►◄►◄►◄► #

PATH_Q1 = "em-M1_dm-M1_e-50_b-48_z-10_lr-0.001_name-urim_auto_encoder_param.pkl"
PATH_Q4 = "em-M1_dm-M1_e-100_b-48_z-10_lr-0.0005_name-urim_auto_encoder_param_method-5.pkl"

PATH_TO_MAN_IMAGE = "man.png"

DEFAULT_ITERATIONS = 10000
DEFAULT_RESTORE_TYPE = "denoising"
# DEFAULT_RESTORE_TYPE = "inpainting"

# DEBUG_MODE = True
DEBUG_MODE = False

# in case there is no more cuda memory left and we still want to train, set the flag to true
# CPU_MODE = True
CPU_MODE = False

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#

def get_mask_noise(std):
  '''returns a curry function that only takes a vector and the std value is set'''
  def add_normal_noise(tensor):
    return torch.clip(tensor + to_cuda(torch.randn(tensor.size())) * std,-1,1)
  return add_normal_noise

def inpainting(image, square_grid, minV):
  '''given a grid and a min value, sets that value across the grid indices to the image'''
  corrupt_image = image.clone()
  corrupt_image[:,:,square_grid[0],square_grid[1]] = minV
  return corrupt_image

def denoising(corrupt_image, std, ae, iterations, saving_name, subdir):
  '''restores a noisy image'''

  save_tensor_im(corrupt_image, SavingPath.get_path(f"{saving_name}_corrupt_image.png", subdir))

  # since according to the DIP algorithm the z vector needs to be fixed, we first get the z vector 
  # so we may later verify by assertion that it has not chainged!
  z1, restored_im = ae(corrupt_image)

  losses = [] # save the losses of each iteration
  mse = nn.MSELoss() 
  progress = Progress(iterations,1) # set the user fedback

  # since we want to fix the z vector, we only optimize the decoders parameters!
  optimizer = torch.optim.SGD(ae.decoder.parameters(), lr=0.0005, momentum=0.9)
  for iter in range(iterations):

    optimizer.zero_grad()

    z2, restored_im = ae(corrupt_image)
    assert(torch.sum(z2.view(-1)-z1.view(-1)) == 0) # assert the returned z has not chainged!

    # likelyhood term:
    loss_likelyhood = mse(restored_im, corrupt_image)
    # regularization term: the prior needs to be independent from the corrupt_image input - the AE still generates the same image as the input
    # assuming the original image I=restored_im, therefore, I=AE(I). hence minimize its error
    loss_prior = mse(restored_im, ae(restored_im)[1])
    
    # loss = loss_likelyhood + loss_prior
    loss = loss_likelyhood + loss_prior*(2*std**2)

    losses.append(loss.cpu().detach().item())
    loss.backward()
    optimizer.step()

    # save images and print feedback
    if(progress.isNextPercent(iter)):
      save_tensor_im(restored_im, SavingPath.get_path(f"{saving_name}_restored_im.{iter}.png", subdir))
      print(f"{subdir} | Epoch:{iter}/{iterations} | progress:{progress.getCurrentProgress(iter)} | loss={loss:.5f} | loss_prior={loss_prior:.5f} | loss_likelyhood={loss_likelyhood:.5f}")

  save_loss_plot(losses, None, subdir)

def restore_inpainting(corrupt_image, ae, iterations, square_grid, minV, saving_name, subdir):
  '''restores a corrupt image from inpainting'''

  save_tensor_im(corrupt_image, SavingPath.get_path(f"{saving_name}_corrupt_image.png", subdir))

  # since according to the DIP algorithm the z vector needs to be fixed, we first get the z vector 
  # so we may later verify by assertion that it has not chainged!
  z1, _ = ae(corrupt_image)

  losses = [] # save the losses of each iteration
  mse = nn.MSELoss() 
  progress = Progress(iterations,1) # set the user fedback

  # since we want to fix the z vector, we only optimize the decoders parameters!
  optimizer = torch.optim.SGD(ae.decoder.parameters(), lr=0.0005, momentum=0.9)

  for iter in range(iterations):

    optimizer.zero_grad()

    z2, restored_im = ae(corrupt_image)
    assert(torch.sum(z2.view(-1)-z1.view(-1)) == 0) # assert the returned z has not chainged!

    # likelyhood term, but first applying the inpainting
    loss_likelyhood = mse(inpainting(restored_im, square_grid, minV), inpainting(corrupt_image, square_grid, minV))
    # regularization term: the prior needs to be independent from the corrupt_image input - the AE still generates the same image as the input
    # assuming the original image I=restored_im, therefore, I=AE(I). hence minimize its error
    loss_prior = mse(restored_im, ae(restored_im)[1])
    
    loss = loss_likelyhood + loss_prior*0.005

    losses.append(loss.cpu().detach().item())
    loss.backward()
    optimizer.step()

    # save images and print feedback
    if(progress.isNextPercent(iter)):
      save_tensor_im(restored_im, SavingPath.get_path(f"{saving_name}_restored_im.{iter}.png", subdir))
      print(f"{subdir} | Epoch:{iter}/{iterations} | progress:{progress.getCurrentProgress(iter)} | loss={loss:.5f} | loss_prior={loss_prior:.5f} | loss_likelyhood={loss_likelyhood:.5f}")
  
  save_loss_plot(losses, None, subdir)

def to_cuda(*args):
  ''' function to easily pass tensors to the cuda device'''
  if(torch.cuda.is_available() and not CPU_MODE):
    cuda_list = []
    for arg in args:
      cuda_list.append(arg.cuda())
    if(len(cuda_list) == 1): return cuda_list[0]
    elif(len(cuda_list) == 0): Exception(f"At least one argument needs to be sent")
    return tuple(cuda_list)
  else:
    args = list(args)
    if(len(args) == 1): return args[0]
    elif(len(args) == 0): Exception(f"At least one argument needs to be sent")
    else: return tuple(args)

def load_ae(path):
  '''returns the AE from the saved path'''
  file_args = read_args_from_file(path)
  ae = AE(file_args["e_model"], file_args["d_model"], file_args["z"])
  ae.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
  ae.encoder.eval()
  ae.decoder.train()
  return to_cuda(ae)

def main():
  
  torch.autograd.set_detect_anomaly(True) # set to true to help debug

  # get CLI args
  parser = argparse.ArgumentParser(description='Ex3_5')
  parser.add_argument('-i', '--iterations', type=int, default=DEFAULT_ITERATIONS, metavar=('iterations'),
                      help=f'Number of iterations to restore the image (default: {DEFAULT_ITERATIONS})')
  parser.add_argument('--std', type=float, default=0.1, metavar=('std'),
                      help=f'Set the std in case of denoising (default: {0.1})')
  parser.add_argument('-t', '--type', type=str, default=DEFAULT_RESTORE_TYPE, metavar=('question path'),
                        help=f'Sets if to do inpainting restoration or denoising restoration (default: {DEFAULT_RESTORE_TYPE})')
  parser.add_argument('-im', '--image_type', type=str, default="normal", metavar=('image type'),
                        help=f'Sets the kind of images to use. Options are: "normal","flip","negative","const","const_noise","man". (default: "normal")')
  parser.add_argument('--str', type=str, default="", metavar=('str'),
                        help=f'Appends a string to the end of the path name. (default: "")')
  args = parser.parse_args()

  # initialization
  init_randomness()
  init_flags(CPU_MODE, DEBUG_MODE)

  # set parameters according to input args
  if(DEBUG_MODE): SavingPath(f"Debug_name-{socket.gethostname()}")
  else: SavingPath(f'restoring_{args.type}_name-{socket.gethostname()}')

  # simple assert that models trained from Q1 and Q4 where trained with the same batch size
  file_args_m1 = read_args_from_file(PATH_Q1)
  file_args_m4 = read_args_from_file(PATH_Q4)
  assert(file_args_m1["batch"] == file_args_m4["batch"])

  # set parameters according to input args
  dataset, _ = get_dataset(file_args_m1["batch"])

  # to get random images for the testing, set the random seed and then set it back 
  random.seed(datetime.now())
  image = to_cuda(random.choice(get_data_as_list(dataset, 10)))
  original_image = image
  # set back the random seed
  init_randomness()

  # == create the image according by the type == #

  # switches according if to run on a normal image from the dataset or an outlier image
  if(args.image_type == "normal"):
    pass # if it tis a normal image, nothing to be done. (this is not to fall into the exception)
  elif(args.image_type == "flip"):
    image = torch.flip(image,[3])
  elif(args.image_type == "const"):
    image = image*0+0.5
  elif(args.image_type == "const_noise"):       
    image = image*0+0.5
    image = torch.clip(image + to_cuda((torch.randn(image.size())-0.5)) * 0.5,-1,1)
  elif(args.image_type == "negative"):
    image = image*(-1)
  elif(args.image_type == "man"):
      man = Image.open(PATH_TO_MAN_IMAGE)
      man = (np.asarray(man)/255 - 0.5)/0.5 # this is the transform applied to MNIST by torch
      image[0,0,:,:] = torch.as_tensor(man)
  else: raise Exception(f"No such image type available: {args.image_type}")

  SavingPath.append(f"_imType-{args.image_type}")
  if(args.str != ""): SavingPath.append(f"_{args.str}")

  save_tensor_im(image, SavingPath.get_path(f"_original_image.png"))
  ae = load_ae(PATH_Q4)
  _, restored_im = ae(image)
  save_tensor_im(restored_im, SavingPath.get_path(f"_AE(original_image)_Q4.png"))
  del ae
  ae = load_ae(PATH_Q1)
  _, restored_im = ae(image)
  save_tensor_im(restored_im, SavingPath.get_path(f"_AE(original_image)_Q1.png"))
  del ae

  # == run the restoring method according to type or restoring == #

  if(args.type.lower() == "denoising"):
    mask = get_mask_noise(args.std) # get the curry function to works as the corrupting mask
    corrupt_image = to_cuda(mask(image)) # get the corrupted image
    # run on model from Q4
    ae = load_ae(PATH_Q4) 
    denoising(corrupt_image, args.std, ae, args.iterations, f"_Q4_denoising_std-{args.std}", "Q4")
    # run on model from Q4
    ae = load_ae(PATH_Q1)
    denoising(corrupt_image, args.std, ae, args.iterations, f"_Q1_denoising_std-{args.std}", "Q1")

  elif(args.type.lower() == "inpainting"):
    # creating a random windom in the middle of the image to be set of to the min value of the image - that should be -1
    # but depends if the image was inverted.
    mask_size = 8
    # we set the patch to only be located at the center of the image (at random). if not, many times it did not affect as 
    # it randomly picked part of the image  that was already black
    top_left_corner_row = np.random.randint(int(image.shape[2]//2) - mask_size, size=1) + int(image.shape[2]//4)
    top_left_corner_col = np.random.randint(int(image.shape[3]//2) - mask_size, size=1) + int(image.shape[3]//4)
    square_grid = np.meshgrid(np.arange(top_left_corner_row, top_left_corner_row + mask_size), np.arange(top_left_corner_col, top_left_corner_col + mask_size))

    minV = original_image.min() # get the min value for setting in the inpainting window
    #if the images is its negative, set the min value to the max of the original image
    if(args.image_type == "negative"): minV = original_image.max() 

    corrupt_image = to_cuda(inpainting(image, square_grid, minV)) # get the corrupted image
    # run on model from Q4
    ae = load_ae(PATH_Q4)
    restore_inpainting(corrupt_image, ae, args.iterations, square_grid, minV, f"_Q4_inpainting", "Q4")
    # run on model from Q4
    ae = load_ae(PATH_Q1)
    restore_inpainting(corrupt_image, ae, args.iterations, square_grid, minV, f"_Q1_inpainting", "Q1")
    
  else: raise Exception(f"No such type available: {args.type.lower()}")

  # wrap up
  SavingPath.finish("-Done")
  print(f"Current dir path:\n{SavingPath.get_dir_path()}")

if __name__ == '__main__':
  # change the current directory to the location of the file to read the model and save images and plots in the right location
  os.chdir(os.path.dirname(os.path.realpath(__file__)))
  torch.backends.cudnn.enabled = True
  torch.backends.cudnn.benchmark =True
  main()