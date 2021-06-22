import os
import time
import numpy as np
from itertools import cycle
from PIL import Image
import os
import sys
import time
import torch
import socket
import random
import numpy as np
from PIL import Image
from itertools import cycle
from datetime import datetime
import matplotlib.pyplot as plt
from contextlib import redirect_stdout
import time

class SavingPath():
  ''' class to creates the correct saving path '''

  def __init__(self, name, saveDirName = "Results"):
    SavingPath._name = name

    dir_path = os.path.dirname(os.path.realpath(__file__))
    SavingPath._saveDir = os.path.join(dir_path, saveDirName, f"{time.strftime('%b-%d_%H.%M.%S', time.localtime())}_{name}")

    if not os.path.exists(SavingPath._saveDir):
      os.makedirs(SavingPath._saveDir)

    print(f"Output dir path is: {SavingPath._saveDir}")

    SavingPath._path = os.path.join(SavingPath._saveDir, SavingPath._name)
    
  @staticmethod
  def get_path(suffix = "", subdir = None):
    ''' A static method that return the correct saving path '''
    if(subdir is None):
      return SavingPath._path + suffix
    else:
      SavingPath.make_subdir(subdir)
      return os.path.join(SavingPath._saveDir, subdir, SavingPath._name) + suffix

  @staticmethod
  def append(suffix = ""):
    SavingPath.finish(suffix)

  @staticmethod
  def finish(suffix = ""):
    os.rename(SavingPath._saveDir, SavingPath._saveDir+suffix)
    SavingPath._saveDir = SavingPath._saveDir+suffix
    SavingPath._path = os.path.join(SavingPath._saveDir, SavingPath._name)

  @staticmethod
  def get_dir_path():
    return SavingPath._saveDir

  @staticmethod
  def make_subdir(name):
    subdir = os.path.join(SavingPath._saveDir, name)
    if not os.path.exists(subdir):
      os.makedirs(subdir) 

class Progress:
  '''Class to give the user feedback on the process adn expected time'''
  def __init__(self, total, percent_step = 10):
    self.total = total
    self.percent_step = percent_step
    self.start_time = time.time()
    self.last_percent = 0

  def getETA(self, current):
    current += 1
    eta_secs = ((self.total - current)*(time.time() - self.start_time))/current
    current_percent = int(current/self.total * 100)
    return f"{current_percent}% - {current}/{self.total} - ETA: {int((eta_secs//60)%60):.0f}:{int(eta_secs%60):02d} [m:s]"

  def getPercent(self, current):
    return int(current/self.total * 100)

  def isNextPercent(self, current):
    current_percent = int(current/self.total * 100)
    old_percent = int((current-1)/self.total * 100)
    return current_percent != old_percent

  def getCurrentProgress(self, current):
    return int(current/self.total * 100)

  def printProgress(self, current, prefix = ""):
    current_percent = int(current/self.total * 100)
    if(self.last_percent + self.percent_step <= current_percent):
      self.last_percent = current_percent
      eta_secs = ((100 - current_percent)*(time.time() - self.start_time))/current_percent
      print(f"{prefix}. Progress: {current_percent}% - ETA: {eta_secs//60:.0f}:{int(eta_secs%60):02d} [m:s]")
    
    if(current == self.total - 1):
      print("100%")

  def printTotalRunningTime(self):
    print("Total running time: %s[sec] " % round((time.time() - self.start_time), 2))

class Running_Time:
  '''Class that stores the initialized time and returns the running time'''

  def init():
    Running_Time.start_time = time.time()

  def get_running_time():
    current_time = time.time()
    return f"{((current_time - Running_Time.start_time)//60):.0f}:{(current_time - Running_Time.start_time)%60:.0f} min:sec"

class RunInterval:
  '''Class that wraps the condition of executing every some set time '''

  def __init__(self, secs = 0, minuts = 0, hours = 0) -> None:
      self.interval_secs = secs + minuts*60 + hours*60*60
      self.start_time = time.time()

  def can_execute(self):
    if(time.time() - self.start_time < self.interval_secs): return False
    self.start_time = time.time()
    return True

  def try_execute(self, action):
    if(time.time() - self.start_time < self.interval_secs): return False
    self.start_time = time.time()
    action()
    return True

def print_parameter(name, val):
    print(f"{name}: {val}")

def save_tensor_im(tensor, path):
  '''saves an image (can be with 3 channels) stored in a tenstor. expects the tensor to be 3 o 4 dimensions'''
  if(len(tensor.shape) == 4):
    save_im(tensor.cpu().detach().numpy().reshape((tensor.shape[1], tensor.shape[2], tensor.shape[3])), path)
  elif(len(tensor.shape) == 3):
    save_im(tensor.cpu().detach().numpy().reshape((tensor.shape[0], tensor.shape[1], tensor.shape[2])), path)
  else:
    raise Exception("cant save tensor")

def save_im(im, path):
  '''saves an image to file that has its values in the [-1,1] range'''
  img = ((im + 1) * 127.5).astype(np.uint8) 
  img = img.transpose(1, 2, 0)
  scaling = 10 if img.shape[2] == 1 else 5
  scaled_im = img.repeat(scaling, axis=0).repeat(scaling, axis=1) # scale image
  # scaled_im = img
  if(scaled_im.shape[2] == 1):
    scaled_im = scaled_im.reshape(scaled_im.shape[0], scaled_im.shape[1])
  scaled_im = Image.fromarray(scaled_im)
  scaled_im.save(path, "PNG")
  
def get_cycle_colors():
  '''Colors for plots'''
  colors_list = 'brgycm'
  return cycle(colors_list)

def init_randomness():
  '''sets the seed of both the random module and torch to a specific seed for reproducibility'''
  # ◄►◄► Set random seed for reproducibility ◄►◄► #
  manualSeed = 999
  #manualSeed = random.randint(1, 10000) # use if you want new results
  print("Random Seed: ", manualSeed)
  random.seed(manualSeed)
  torch.manual_seed(manualSeed)

def init_flags(CPU_MODE, DEBUG_MODE):
  '''gives the user feedback of the cpu and debug flags and well as if cuda is available in the machine'''
  print("================================================================")
  # ◄►◄► Set CUDA GPU ◄►◄► #
  if(torch.cuda.is_available() and not CPU_MODE):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(device)
    print("is CUDA initialized: "+ "YES" if torch.cuda.is_initialized() else "NO")
  print(f"◄►◄►◄►◄►◄►◄► CPU MODE flag = {CPU_MODE} ◄►◄►◄►◄►◄►◄►")
  print(f"◄►◄►◄►◄►◄►◄► DEBUG MODE flag = {DEBUG_MODE} ◄►◄►◄►◄►◄►◄►")
  print("================================================================")

def save_checkpoint(epoch, model, optimizer, loss_func, epoch_finished = False, subdir = None):
  '''saves a checkpoint of the training which can be loaded later on with the load_checkpoint function'''
  checkpoint = { 
    'epoch': epoch,
    'model': model,
    'optimizer': optimizer,
    'loss_func': loss_func}
  sub_index = 0
  while(True):
    suffix = "epoch-done" if epoch_finished else sub_index
    saving_path = SavingPath.get_path(f"_checkpoint_epoch-{epoch+1}.{suffix}.pkl", subdir)
    sub_index += 1
    if(not os.path.exists(saving_path)): break

  torch.save(checkpoint, saving_path)
  return saving_path

def load_checkpoint(path):
  '''returns a dict with the loaded info from a checkpoint created when using the save_checkpoint function'''
  checkpoint = torch.load(path, map_location=torch.device('cpu'))
  epoch = checkpoint['epoch']
  model = checkpoint['model']
  optimizer = checkpoint['optimizer']
  loss_func = checkpoint['loss_func']
  return (epoch, model, optimizer, loss_func)

# ◄►◄► Losses Plot ◄►◄► #

def show_loss_plot(losses, dataloader_size): general_plot(losses, dataloader_size, False)

def save_loss_plot(losses, dataloader_size, subdir = None): general_plot(losses, dataloader_size, True, subdir)
    
def general_plot(losses, dataloader_size, to_save, subdir = None):
  '''a general function to both show or save plots. it can take either a single or multiple plots which it then 
  plots them all on the same figure. if the losses contain more than a single loss, then the fist value of each loss 
  is supposed to be its label as a sting'''
  if(len(losses) == 0): return
  fig = plt.figure()
  fig.set_size_inches(16, 12)
  if(isinstance(losses[0], list)):
    labels = losses[0]
    losses = losses[1:]
    for loss in map(list, zip(*losses)):
      plt.plot(np.arange(1, len(losses) + 1), loss, alpha=0.8)
    plt.legend(labels)
  else:
    plt.plot(np.arange(1, len(losses) + 1), losses)
    plt.legend(["losses"])
  if(dataloader_size is None):
    plt.xlabel(f"# of iterations")
  else:
    plt.xlabel(f"# of iterations. 1 epoch = {dataloader_size} iterations")
  plt.ylabel("loss")
  if(to_save):
    plt.savefig(SavingPath.get_path(f"plot_iter-{len(losses) + 1}.png", subdir), dpi = 100)
  else:
    plt.show()
  plt.close(fig)


def save_info(args, ae, dataset, loss_func, optimizer, CPU_MODE):
  '''saves the info of the running configuration to file'''
  with open(SavingPath.get_path('_AA_info.txt'), 'a+') as file:
    file.write(f"Running Script:\t\t\t{os.path.realpath(__file__)}\n")
    file.write(f"Creation Time:\t\t\t" + datetime.now().strftime("%d-%m-%Y_%H:%M:%S.%f") + "\n")
    file.write(f"PC Name:\t\t\t\t\t\t{socket.gethostname()}\n")
    file.write(f"Process ID:\t\t\t\t\t{os.getpid()}\n")
    file.write(f"CUDA Available:\t\t\t{torch.cuda.is_available()}\n")
    file.write(f"CUDA Initialized:\t\t{torch.cuda.is_initialized()}\n")
    file.write(f"CPU_MODE:\t\t\t\t\t\t{CPU_MODE}\n")
    file.write(f"DEBUG_MODE:\t\t\t\t\t{args.debug}\n")
    file.write(f"Epochs:\t\t\t\t\t\t\t{args.epochs}\n")
    file.write(f"Z-Size:\t\t\t\t\t\t\t{args.z_size}\n")
    file.write(f"Batches:\t\t\t\t\t\t{args.batch_size}\n")
    file.write(f"Encoder Params:\t\t\t{ae.encoder.count_parameters()}\n")
    file.write(f"Decoder Params:\t\t\t{ae.decoder.count_parameters()}\n")
    file.write(f"Dataset Len:\t\t\t\t{len(dataset)}\n")
    file.write(f"Output Path:\t\t\t\t{SavingPath.get_dir_path()}\n")
    file.write(f"Appended Str:\t\t\t\t'{args.str}'\n")
    file.write(f"Loss Function:\t\t\t{loss_func}\n")
    file.write(f"Encoder Name:\t\t\t\t{args.gen_model}\n")
    file.write(f"Decoder Name:\t\t\t\t{args.dis_model}\n")
    file.write("\n")
    file.write(f"Optimizer: {optimizer}\n")
    file.write("\n")
    file.write(f"Decoder: {ae.encoder}\n")
    file.write("\n")
    file.write(f"Encoder: {ae.decoder}\n")
    file.write("\n")
    with redirect_stdout(file):
      file.write("Encoder:\n")
      data_shape = next(iter(dataset))[0].shape
      ae.encoder.summary((args.batch_size, data_shape[0], data_shape[1], data_shape[2]), not CPU_MODE)
      file.write("\n")
      file.write("Decoder:\n")
      ae.decoder.summary((args.batch_size, args.z_size), not CPU_MODE)

def check_cli(cli, ae, epoch, losses, dataloader, optimizer, loss_func):
  '''parses the cli entered by the user'''
  if(cli == "exit" or cli == "stop"):
    print("exiting")
    SavingPath.finish("-Stopped")
    print(f"Current dir path:\n{SavingPath.get_dir_path()}")
    sys.exit()
  elif(cli == "save"):
    torch.save(ae.state_dict(), SavingPath.get_path(f"_auto_encoder_param-{epoch+1}.pkl"))
    print("models saved")
  elif(cli == "plot"):
    show_loss_plot(losses, len(dataloader))
    print("plot shown")
  elif(cli == "save_loss"):
    save_loss_plot(losses, len(dataloader))
    print("plot saved")          
  elif(cli == "pause"):
    print("Pausing.. Waiting for user input (press enter)")
    new_cli = input()
    print("User input entered. Resuming..")     
  elif(cli == "info"):
    print(f"PC Name: {socket.gethostname()}")
    print(f"PID: {os.getpid()}")
    print(f"encoder: {ae.encoder.model_name}")
    print(f"decoder: {ae.decoder.model_name}")
    print(f"Current dir path:\n{SavingPath.get_dir_path()}")
  elif(cli == "checkpoint"):
    saving_path = save_checkpoint(epoch, ae, optimizer, loss_func)
    print(f"Created a checkpint at {saving_path}")
  elif(cli == "help"):
    print("Enter:\texit\n\tsave\n\tplot\n\tsave_loss\n\tpause\n\tinfo\n\tcheckpoint\n\thelp\n")
  else:
    print("Unrecognized command. Enter 'help' for valid commands.")