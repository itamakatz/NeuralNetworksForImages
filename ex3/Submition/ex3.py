import os
import torch
import socket
import argparse
import matplotlib
import sys, select
import numpy as np 
matplotlib.use('Agg')
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from scipy.stats import kurtosis
from itertools import combinations

from models import AE
from common import get_dataset
from utils import SavingPath, print_parameter, save_tensor_im, Progress, init_randomness, init_flags, save_loss_plot, save_info, check_cli, get_cycle_colors

os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # needed in case of conflict error with numpy

# ◄►◄►◄►◄►◄► Magic Variables ◄►◄►◄►◄►◄► #

#  see the CLI args in the main method to see the description of each variable

DEFAULT_Z = 10
DEFAULT_BATCH = 48
DEFAULT_EPOCHS = 100
DEFAULT_FIT_Z = True
DEFAULT_E_MODEL = "M1"
DEFAULT_D_MODEL = "M1"
DEFAULT_LEARNING_RATE = 0.5e-3
# DEFAULT_SEARCH_MODE = True
DEFAULT_SEARCH_MODE = False
DEFAULT_OPTIMIZER = "SGD"
DEFAULT_APPEND_STR = ""

# ---- #

# this will overwrite the parameters in case of debug mode
DEBUG_MODE_EPOCHS = 2
DEBUG_MODE_BATCH = 48 #48 #96
DEBUG_MODE_PATH = "Debugging"

# ---- #

# DEFAULT_DEBUG_MODE = True
DEFAULT_DEBUG_MODE = False

# in case there is no more cuda memory left and we still want to train, set the flag to true
# CPU_MODE = True 
CPU_MODE = False

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#

def kl_loss(p, q):
  '''custom implementation of the KL Divergence'''
  p,q = p.view(p.numel()), q.view(q.numel())
  p,q = F.softmax(p, dim=0), F.softmax(q, dim=0)
  p,q = to_cuda(p,q)
  s1 = torch.sum(p*torch.log(p/q))
  s2 = torch.sum((1-p)*torch.log((1-p)/(1-q)))

  return s1+s2

def mse(x,y):
  '''custom flavor of the mse loss'''
  return torch.sqrt(torch.sum((torch.abs(x-y))**2))

def train_auto_encoder(ae, num_epochs, loss_func, optimizer, dataloader, fit_z, searching_mode = False):
  '''training of the AE '''
  losses = list() # to store the loss of each batch
  if(fit_z): losses.append(["total_loss", "batch_loss", "sparsity_loss"]) # in case of multiple losses, adds their labels 
  kdLoss = nn.KLDivLoss(reduction='batchmean') # the original KL Divergence loss used in one of the methods below
  epoch = 0

  while epoch < num_epochs:
    progress = Progress(len(dataloader), 1) # sets the feedback for the user
    
    # train
    ae.train() 
    for batch_count, data in enumerate(dataloader):
      
      # linux user can have non blocking input to dynamically request plots, pause, save etc.
      if(os.name == "posix"): 
        has_input, o, e = select.select( [sys.stdin], [], [], 0.1 ) 
      else:
        has_input = False

      data = data[0]
      if (torch.cuda.is_available() and not CPU_MODE): data = data.cuda()
      optimizer.zero_grad()
      z, outputs = ae(data) # get the AE prediction
      batch_loss = loss_func(outputs.type(torch.FloatTensor), data.type(torch.FloatTensor)) # calculate the AE loss

      if(fit_z): # for Q4, if we would like to force z to have a particular distribution
        
        method = 9 # method 5 is the best we achieved.

        if(method == 1): # centers the scattering to a normal like shape
          sparsity_loss = kl_loss(torch.as_tensor(np.ones(z.shape[1])*0.01), torch.sum(z, dim=0, keepdim=True))
          total_loss = 5*batch_loss+sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        elif(method == 2):
          var_batch = torch.var(z, dim=0) # calc the variance across the batch
          var_batch_loss = mse(torch.mean(var_batch), 1/12) 

          mean_z = torch.mean(z, dim=1) # calc the mean across the z vector
          mean_z_loss = mse(torch.mean(mean_z), 0.5)

          kur_batch = to_cuda(torch.as_tensor(kurtosis(z.cpu().detach().numpy(), axis=0, fisher=False))) # calc the kurtosis across the batch
          kur_batch_loss = mse(torch.mean(kur_batch), 9/5)

          center_loss = mse(z-0.5, to_cuda(torch.zeros_like(z))) # general additional loss to force to stay close to the  [0.5]^d

          sparsity_loss = mean_z_loss*0 + var_batch_loss*10*dataloader.batch_size/96 + kur_batch_loss + center_loss*00.1
          total_loss = batch_loss+100*sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        elif(method == 3):
          var_batch = torch.var(z, dim=0) # calc the variance across the batch
          var_batch_loss = mse(var_batch, 1/12*to_cuda(torch.ones_like(var_batch)))

          mean_z = torch.mean(z, dim=1) # calc the mean across the z vector
          mean_z_loss = mse(mean_z, 0.5*to_cuda(torch.ones_like(mean_z)))

          kur_batch = to_cuda(torch.as_tensor(kurtosis(z.cpu().detach().numpy(), axis=0, fisher=False))) # calc the kurtosis across the batch
          kur_batch_loss = mse(kur_batch, 9/5*to_cuda(torch.ones_like(kur_batch)))

          center_loss = mse(z-0.5, to_cuda(torch.zeros_like(z))) # general additional loss to force to stay close to the  [0.5]^d

          sparsity_loss = mean_z_loss*0 + var_batch_loss*10*dataloader.batch_size/96 + kur_batch_loss + center_loss*00.1
          total_loss = batch_loss+100*sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])
          
        elif(method == 4):
          uniform_v = to_cuda(torch.rand(z.shape)) # generate a vector from the normal distribution and compute the loss according to that
          uniform_loss = mse(z, uniform_v)

          var_batch = torch.var(z, dim=0) # calc the variance across the batch
          var_batch_loss = mse(var_batch, 1/12*to_cuda(torch.ones_like(var_batch)))

          mean_z = torch.mean(z, dim=1) # calc the mean across the z vector
          mean_z_loss = mse(mean_z, 0.5*to_cuda(torch.ones_like(mean_z)))

          kur_batch = to_cuda(torch.as_tensor(kurtosis(z.cpu().detach().numpy(), axis=0, fisher=False))) # calc the kurtosis across the batch
          kur_batch_loss = mse(kur_batch, 9/5*to_cuda(torch.ones_like(kur_batch)))

          center_loss = mse(z-0.5, to_cuda(torch.zeros_like(z))) # general additional loss to force to stay close to the  [0.5]^d

          sparsity_loss = mean_z_loss*0 + var_batch_loss*10*dataloader.batch_size/96 + kur_batch_loss + center_loss*0 + uniform_loss*00.1
          
          total_loss = batch_loss+100*sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        elif(method == 5):
          var_batch = torch.var(z, dim=0) # calc the variance across the batch
          var_batch_loss = mse(var_batch, 1/12*to_cuda(torch.ones_like(var_batch)))

          kur_batch = to_cuda(torch.as_tensor(kurtosis(z.cpu().detach().numpy(), axis=0, fisher=False))) # calc the kurtosis across the batch
          kur_batch_loss = mse(kur_batch, 9/5*to_cuda(torch.ones_like(kur_batch)))

          sparsity_loss = var_batch_loss*10 + kur_batch_loss
          
          total_loss = batch_loss+0.1*sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        elif(method == 6):
          var_batch = torch.var(z, dim=0) # calc the variance across the batch
          var_batch_loss = mse(var_batch, 1/12*to_cuda(torch.ones_like(var_batch)))

          kur_batch = to_cuda(torch.as_tensor(kurtosis(z.cpu().detach().numpy(), axis=0, fisher=False))) # calc the kurtosis across the batch
          kur_batch_loss = mse(kur_batch, 9/5*to_cuda(torch.ones_like(kur_batch)))

          sparsity_loss = var_batch_loss*10 + kur_batch_loss
          
          total_loss = batch_loss+0.15*sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        elif(method == 7):
          var_batch = torch.var(z, dim=0) # calc the variance across the batch
          var_batch_loss = mse(var_batch, 1/12*to_cuda(torch.ones_like(var_batch)))

          kur_batch = to_cuda(torch.as_tensor(kurtosis(z.cpu().detach().numpy(), axis=0, fisher=False))) # calc the kurtosis across the batch
          kur_batch_loss = mse(kur_batch, 9/5*to_cuda(torch.ones_like(kur_batch)))

          sparsity_loss = var_batch_loss*10 + kur_batch_loss
          
          total_loss = batch_loss+0.05*sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        elif(method == 8):
          # use the normal KL Divergence loss 
          p = torch.softmax(z, dim=1) # do softmax across the z vector
          p = torch.log(p)
          q = to_cuda(torch.ones_like(p))*5
          q = torch.softmax(q, dim=1) # do softmax across the z vector
          sparsity_loss  = kdLoss(p,q)
          total_loss = batch_loss+sparsity_loss

          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        elif(method == 9):
          # mean_batch = torch.mean(z, dim=0) # calc the mean across the batch
          # mean_batch_loss = mse(mean_batch, 0.5*to_cuda(torch.ones_like(mean_batch)))
          
          mean_z = torch.mean(z, dim=1) # calc the mean across the z vector
          mean_z_loss = mse(mean_z, 0.5*to_cuda(torch.ones_like(mean_z)))

          var_batch = torch.var(z, dim=0) # calc the variance across the batch
          var_batch_loss = mse(var_batch, 1/12*to_cuda(torch.ones_like(var_batch)))

          kur_batch = to_cuda(torch.as_tensor(kurtosis(z.cpu().detach().numpy(), axis=0, fisher=False))) # calc the kurtosis across the batch
          kur_batch_loss = mse(kur_batch, 9/5*to_cuda(torch.ones_like(kur_batch)))

          sparsity_loss = var_batch_loss*10 + kur_batch_loss + mean_z_loss*0.01
          
          total_loss = batch_loss+0.1*sparsity_loss
          print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f} | b_loss={batch_loss:.5f} | sparsity_loss={sparsity_loss:.5f}")
          losses.append([total_loss.cpu().detach().item(), batch_loss.cpu().detach().item(), sparsity_loss.cpu().detach().item()])

        else:
          raise Exception("no such method")
      else:
        # if we are not fitting the latent space we enter here
        total_loss = batch_loss
        print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={total_loss:.5f}")
        losses.append(total_loss.cpu().detach().item())
      
      total_loss.backward()
      optimizer.step()
      
      # if in searching mode, we save images, plots and scattering every 20%
      if (searching_mode and progress.isNextPercent(batch_count) and progress.getPercent(batch_count)%20 == 0):
        save_loss_plot(losses, len(dataloader))
        save_scattering(z, epoch, batch_count)
        for i in range(5):
          save_tensor_im(outputs[i], SavingPath.get_path(f"_epoch-{epoch + 1}_batch_count-{batch_count}.{i}.png"))
      
      # enables CLI input during the training 
      if (has_input):
        cli = sys.stdin.readline().strip()
        cli = cli.lower()
        if(cli.startswith("add") or cli.startswith("remove")): #add or removes number of epochs
          amount = [int(s) for s in cli.split() if s.isdigit()]
          if(len(amount) == 1):
            amount = amount[0]
            old_num_epochs = num_epochs
            num_epochs = num_epochs + amount if cli.startswith("add") else num_epochs - amount
            print(f"Old epochs={old_num_epochs}. New epochs={num_epochs}")
          else:
            print(f"Incorrect syntax. Please try again")  
        if(cli=="save_scatter"):
          save_scattering(z, epoch, batch_count)
          print(f"Saved scatterring")
        else:
          check_cli(cli, ae, epoch, losses, dataloader, optimizer, loss_func)
        
    # free the GPU to avoid memory leak
    del batch_loss
    del total_loss
    if(fit_z):
      del sparsity_loss        
    torch.cuda.empty_cache()

    # save plots and images in case we are not in a seaching mode where many are saved during a single epoch
    if(not searching_mode): # if we are saving on every iteration, no need to to save the batch
      for i in range(len(outputs)):
        save_tensor_im(outputs[i], SavingPath.get_path(f"_epoch-{epoch+1}.{i}.png"))
      save_scattering(z, epoch, batch_count)

    save_loss_plot(losses, len(dataloader))

    epoch += 1
    
  # save the the trained AE
  torch.save(ae.state_dict(), SavingPath.get_path("_auto_encoder_param.pkl"))

# ◄►◄► Prints ◄►◄► #

def save_scattering(z, epoch, batch_count):
  '''saves a plot of the scattering to a file'''
  combs = list(combinations(list(range(0,z.shape[1])), 2))
  colors = get_cycle_colors()
  for i in range(5):
    fig = plt.figure()
    plt.scatter(z.cpu().detach().numpy()[:,combs[i][0]], z.cpu().detach().numpy()[:,combs[i][1]], c=next(colors), s=5, alpha=1)
    plt.xlim(0, 1)
    plt.ylim(0, 1)  
    plt.savefig(SavingPath.get_path(f"scatter_epoch-{epoch + 1}.batch_count-{batch_count}.{i}_dim1-{combs[i][0]}_dim2-{combs[i][1]}.png"), dpi = 100)
    plt.close(fig)

  fig = plt.figure()
  for comb in combs:
    plt.scatter(z.cpu().detach().numpy()[:,comb[0]], z.cpu().detach().numpy()[:,comb[1]], s=5, alpha=0.5)
  plt.xlim(0, 1)
  plt.ylim(0, 1)  
  plt.savefig(SavingPath.get_path(f"scatter_all_epoch-{epoch + 1}.batch_count-{batch_count}.{i}.png"), dpi = 100)
  plt.close(fig)
  

def print_parameters(num_epochs, loss_func, dataloader, lr, z_size):
  '''print the current runing configuration to the user'''
  print("◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►")
  print_parameter("num_epochs", num_epochs)
  print_parameter("im_size", "28x28")
  print_parameter("latent_space", z_size)
  print_parameter("lr", lr)
  print_parameter("batch_size", dataloader.batch_size)
  print_parameter("loss_func", loss_func)
  print("◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►")

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

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

def main():

  torch.autograd.set_detect_anomaly(True) # set to true to help debug

  # ◄►◄► Set CLI ◄►◄► #
  parser = argparse.ArgumentParser(description='Ex3')
  
  parser.add_argument('-e', '--epochs', type=int, default=DEFAULT_EPOCHS, metavar=('epochs'),
                      help=f'Number of epochs to train (default: {DEFAULT_EPOCHS})')
  parser.add_argument('-z', '--z-size', type=int, default=DEFAULT_Z, metavar=('latent_space'),
                      help=f'Latent space size (default: {DEFAULT_Z})')
  parser.add_argument('-b', '--batch', type=int, default=DEFAULT_BATCH, metavar=('batch'),
                      help=f'Batch size (default: {DEFAULT_BATCH})')
  parser.add_argument('-me', '--e-model', type=str, default=DEFAULT_E_MODEL, metavar=('model'),
                      help=f'Sets the model of the encoder. Syntax: Mx (default: "{DEFAULT_E_MODEL}")')
  parser.add_argument('-md', '--d-model', type=str, default=DEFAULT_D_MODEL, metavar=('model'),
                      help=f'Sets the model of the decoder. Syntax: Mx (default: "{DEFAULT_D_MODEL}")')
  parser.add_argument('-lr', type=float, default=DEFAULT_LEARNING_RATE, metavar=('rate'),
                      help=f'Sets the learning rate. (default: {DEFAULT_LEARNING_RATE})')
  parser.add_argument('--debug', action='store_true', default=DEFAULT_DEBUG_MODE,
                      help=f'Switches to debug mode which overwrites other given arguments except the loss function (default: {DEFAULT_DEBUG_MODE})')
  parser.add_argument('--release', action='store_true', default=False,
                      help=f'Overwrites the debug mode flag (default: {False})')
  parser.add_argument('--fit_z', action='store_true', default=DEFAULT_FIT_Z,
                      help=f'Flag if to fit the latent vector z to the uniform distribution (default: {DEFAULT_FIT_Z})')
  parser.add_argument('-sm','--search-mode', action='store_true', default=DEFAULT_SEARCH_MODE,
                      help=f'Switches to testing mode which saves the loss function plot frequently (default: {DEFAULT_SEARCH_MODE})')
  parser.add_argument('--optim', type=str, default=DEFAULT_OPTIMIZER, metavar=('optimizer'),
                        help=f'Sets the optimizer. options are adam or sgd. (default: {DEFAULT_OPTIMIZER})')
  parser.add_argument('--str', type=str, default=DEFAULT_APPEND_STR, metavar=('str'),
                        help=f'Appends a string to the end of the path name. (default: "{DEFAULT_APPEND_STR}")')
  args = parser.parse_args()

  if(args.release): args.debug = False

  # initializations
  init_randomness()
  init_flags(CPU_MODE, args.debug)

  # if in debug mode overwrite the args
  if(args.debug):
    args.epochs = DEBUG_MODE_EPOCHS
    args.batch = DEBUG_MODE_BATCH
    SavingPath(f"{DEBUG_MODE_PATH}_name-{socket.gethostname()}")
  else:
    SavingPath(f"em-{args.e_model}_dm-{args.d_model}_e-{args.epochs}_b-{args.batch}_z-{args.z_size}_lr-{args.lr}_name-{socket.gethostname()}")

  if(args.fit_z): SavingPath.append("_fit-z")
  if(args.str != ""): SavingPath.append(f"_{args.str}")

  # create parameters from the args
  loss_func = nn.MSELoss()
  (dataset, im_size) = get_dataset(args.batch)

  ae = to_cuda(AE(args.e_model, args.d_model, args.z_size))
  
  if(args.optim.lower() == 'adam'): optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr)
  elif(args.optim.lower() == 'sgd'): optimizer = torch.optim.SGD(ae.parameters(), lr=args.lr, momentum=0.9)
  else: raise Exception(f"No such optimizer option named: {args.optim}")

  # save the running configuration to file
  save_info(args, ae, dataset, loss_func, optimizer, CPU_MODE)

  # give user feedback of the current running configuration
  print_parameters(args.epochs, loss_func, dataset, args.lr, args.z_size)

  # start the training
  train_auto_encoder(ae, args.epochs, loss_func, optimizer, dataset, args.fit_z, args.search_mode)

  # wrapping up
  SavingPath.finish("-Done")
  print(f"Current dir path:\n{SavingPath.get_dir_path()}")

if __name__ == '__main__':
  # change the current directory to the location of the file to save images and plots in the right location
  os.chdir(os.path.dirname(os.path.realpath(__file__)))
  main()