import sys, select
import socket
import re
import numpy as np
import argparse
import random
import copy
import os
import matplotlib.pyplot as plt
from scipy.stats import kurtosis
import torch
import torch.optim as optim
# import seaborn as sns

from utils import SavingPath, print_parameter, Progress, get_cycle_colors, \
                  init_randomness, init_flags, check_cli, show_loss_plot, \
                  save_loss_plot, save_tensor_im, save_info, RunInterval, save_checkpoint
from common import count_parameters, get_loss, get_dataset, get_models, get_auto_encoder

# ◄►◄►◄►◄►◄► Magic Variables ◄►◄►◄►◄►◄► #

DEFAULT_EPOCHS = 100
DEFAULT_LOSS = "mse"
DEFAULT_Z = 100
DEFAULT_BATCH = 24#128 #100 # 48
DEFAULT_DATASET = "celeb"
# DEFAULT_DATASET = "MNIST"
DEFAULT_GEN_MODEL = "M5"
DEFAULT_DIS_MODEL = "M5"
# DEFAULT_K = 2 #4
DEFAULT_K = 0
DEFAULT_GENERATOR_STEPS = 1
DEFAULT_DISCRIMINATOR_STEPS = 1
DEFAULT_GEN_LEARNING_RATE = 0.0004 #0.001
DEFAULT_DIS_LEARNING_RATE = 0.0001#0.001
DEFAULT_DEBUG_MODE = True
# DEFAULT_DEBUG_MODE = False

DEFAULT_SEARCH_MODE = True
# DEFAULT_SEARCH_MODE = False

DEBUG_MODE_EPOCHS = 2
DEBUG_MODE_BATCH = 96
DEBUG_MODE_PATH = "Debugging"

NOVEL = False

# CPU_MODE = True
CPU_MODE = False

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#

def train_gan(num_epochs, loss_gen, loss_dis, generator, discriminator, dataloader, im_size, latent_space, datasetName, gen_lr, dis_lr,
        unrolled_steps=3, generator_steps=1, discriminator_steps=1, searching_mode = False):

  # printing the parameters of the training
  print_parameters(num_epochs, loss_gen, generator, discriminator, dataloader, im_size, latent_space, unrolled_steps, generator_steps, discriminator_steps, gen_lr, dis_lr)
  
  losses = list()
  losses.append(["gen_loss", "dis_loss"])
  generator_optimizer = optim.SGD(generator.parameters(), lr=gen_lr, momentum=0.9)
  discriminator_optimizer = optim.SGD(discriminator.parameters(), lr=dis_lr, momentum=0.9)

  epoch = 0
  while epoch < num_epochs:
    progress = Progress(len(dataloader), 1)
    for batch_count, data in  enumerate(dataloader):

      # using this to get user input during execution. Works only on linux
      if(os.name == "posix"):
        has_input, o, e = select.select( [sys.stdin], [], [], 0.1 ) 
      else:
        has_input = False

      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#
      # ◄►◄► Initialization of the data ◄►◄ #
      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#

      # initializing the data
      if datasetName.lower() == "mnist": data = data[0]
      if(torch.cuda.is_available() and not CPU_MODE): data = data.cuda()
      b_size = data.shape[0]
      
      # defining the matrices to compute the loss of the generator and discriminator. 
      # Note that the factor of 0.9 is to avoid making the discriminator overly confident of itself in the case of the celebs dataset.
      response_real = torch.ones((b_size,1))
      response_real = response_real if datasetName.lower() == "mnist" else response_real*0.9
      response_fake = torch.zeros((b_size,1))
      if(torch.cuda.is_available() and not CPU_MODE):
        response_real, response_fake = response_real.cuda(), response_fake.cuda()

      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#
      # ◄►◄► Train Discriminator ◄►◄► #
      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#

      discriminator.train()
      for step in range(discriminator_steps):
        discriminator_optimizer.zero_grad()
        output = discriminator(data)
        real_imgs_loss = loss_dis(output, response_real)
        z = torch.randn((b_size, latent_space), dtype=torch.float)
        if(torch.cuda.is_available() and not CPU_MODE):
          z = z.cuda()
        with torch.no_grad():
          generator_results = generator(z)
        discriminator_generator_results = discriminator(generator_results)
        fake_imgs_loss = loss_dis(discriminator_generator_results, response_fake)
        discriminator_train_loss = real_imgs_loss + fake_imgs_loss
        discriminator_train_loss = discriminator_train_loss.type(torch.FloatTensor)
        discriminator_train_loss.backward()
        discriminator_optimizer.step()

      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄#
      # ◄►◄► Unrolling Training ◄►◄► #
      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄#

      # to avoid mode collapse we train the discriminator for a set amounts of steps before training the generator and then reverting
      # the progress of the discriminator back to the point we deed a deep copy (called backup)
      if unrolled_steps > 0:
        backup = copy.deepcopy(discriminator)
        for step in range(unrolled_steps):
          discriminator_optimizer.zero_grad()
          output = discriminator(data)
          real_imgs_loss = loss_dis(output, response_real)
          z = torch.randn((b_size, latent_space), dtype=torch.float)
          if(torch.cuda.is_available() and not CPU_MODE):
            z = z.cuda()
          with torch.no_grad():
            generator_results = generator(z)
          discriminator_generator_results = discriminator(generator_results)
          fake_imgs_loss = loss_dis(discriminator_generator_results, response_fake)
          discriminator_train_loss = real_imgs_loss + fake_imgs_loss
          discriminator_train_loss = discriminator_train_loss.type(torch.FloatTensor)
          discriminator_train_loss.backward()
          discriminator_optimizer.step()

      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#
      # ◄►◄► Train Generator ◄►◄► #
      # ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#
      
      generator.train()
      generator_optimizer.zero_grad()
      for step in range(generator_steps):
        z = torch.randn((b_size, latent_space))
        if(torch.cuda.is_available() and not CPU_MODE):
          z = z.cuda()
        generator_results = generator(z)
        discriminator_generator_results = discriminator(generator_results)
        generator_train_loss = loss_gen(discriminator_generator_results, response_fake)
        generator_train_loss.backward()
        generator_optimizer.step()

      # The last part if the unrolled steps implementations - revert to the previous stated of the discriminator as if it did not train!
      if unrolled_steps > 0:
        discriminator.load(backup)
        del backup

      # enables CLI input during the training 
      if (has_input):
        cli = sys.stdin.readline().strip()
        cli = cli.lower()
        if(cli == "exit"):
          print("exiting")
          SavingPath.finish("-Stopped")
          print(f"Current dir path:\n{SavingPath.get_dir_path()}")
          sys.exit()
        elif(cli == "save"):
          torch.save(generator.state_dict(), SavingPath.get_path(f"_generator_param_epoch-{epoch+1}.pkl"))
          torch.save(discriminator.state_dict(), SavingPath.get_path(f"_discriminator_param_epoch-{epoch+1}.pkl"))
          print("models saved")
        elif(cli == "add10"):
          old_num_epochs = num_epochs
          num_epochs += 10
          print(f"Old num_epochs={old_num_epochs}. New num_epochs={num_epochs}")
        elif(cli == "remove10"):
          old_num_epochs = num_epochs
          num_epochs -= 10
          print(f"Old num_epochs={old_num_epochs}. New num_epochs={num_epochs}")
        elif(cli == "add1"):
          old_num_epochs = num_epochs
          num_epochs += 1
          print(f"Old num_epochs={old_num_epochs}. New num_epochs={num_epochs}")
        elif(cli == "remove1"):
          old_num_epochs = num_epochs
          num_epochs -= 1
          print(f"Old num_epochs={old_num_epochs}. New num_epochs={num_epochs}")
        elif(cli == "output_dir" or cli == "dir" or cli == "path"):
          print(f"Current dir path:\n{SavingPath.get_dir_path()}")
        elif(cli == "losses" or cli == "loss" or cli == "show_plot"):
          show_loss_plot(losses, len(dataloader))
          print("Plot shown")
        elif(cli == "save_losses" or cli == "save_loss" or cli == "save_plot"):
          save_loss_plot(losses, len(dataloader))
          print("Plot saved")
        elif(cli == "pause"):
          print("Pausing.. Waiting for user input (press enter)")
          new_cli = input()
          print("User input entered. Resuming..")
        elif(cli == "print_args"):
          print_parameters(num_epochs, loss_gen, generator, discriminator, dataloader, im_size, latent_space, unrolled_steps, generator_steps, discriminator_steps, gen_lr, dis_lr)
        elif(cli == "print_models"):
          print_parameter("generator", generator)
          print_parameter("discriminator", discriminator)
        elif(cli == "help"):
          print("Enter:\nexit: stop execution.\nsave: save the models at this point in time and continue execution.\n"+
              "add10\\add1: add 10\\1 total epochs to run.\nremove10\\remove1: remove 10\\1 epochs.\n"+
              "losses\loss: show plot of the current losses.\nsave_losses\save_loss: shows plot of losses and saves it to an image.\n"+
              "output_dir\\dir\\path: print the output dir path.\npause: pause execution of the training. To continue press enter.\n"+
              "print_args: prints the configuration of the gan learning.\nprint_models: prints the layers of the models")
        else:
          print("Unrecognized command. Enter 'help' for valid commands.")
          
      # appending the losses for the plots
      losses.append([generator_train_loss.cpu().detach().item(), discriminator_train_loss.cpu().detach().item()])
      # printing the progress of the training and the estimated finishing time.
      progress.printProgress(batch_count, f"epoch#: {epoch+1}/{num_epochs}")
      # in case we are interested in saving images and plots in mid-epoch, we enter here
      if(searching_mode and progress.isNextPercent(batch_count)):
        save_loss_plot(losses, len(dataloader))
        # saving only 5 images of the batch
        for i in range(5):
          save_tensor_im(generator_results[i], SavingPath.get_path(f"_epoch-{epoch+1}.batch_count-{batch_count}.{i}.png"))

    # after each epoch print status
    print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | gen_loss={generator_train_loss:.5f} | dis_loss={discriminator_train_loss:.5f}")
    # save plots and images in case we are not in a seaching mode where many are saved inside during a single epoch
    if(not searching_mode): # if we are saving on every iteration, no need to to save the batch
      for i in range(len(generator_results)):
        save_tensor_im(generator_results[i], SavingPath.get_path(f"_epoch-{epoch+1}.{i}.png"))

    # save the plots
    save_loss_plot(losses, len(dataloader))
    epoch += 1

    # free the GPU to avoid memory leak
    del generator_train_loss
    del discriminator_train_loss
    torch.cuda.empty_cache()

  # after the training is over, save the generator and discriminator models
  torch.save(generator.state_dict(), SavingPath.get_path("_generator_param.pkl"))
  torch.save(discriminator.state_dict(), SavingPath.get_path("_discriminator_param.pkl"))

def gauss_details(z):
  '''returns the statistics of the vector z. that is: the mean, variance and kurtosis'''
  mean = torch.mean(z, dim=1)
  var = torch.var(z, dim=1)
  kur = torch.from_numpy(kurtosis(z.cpu().detach().numpy(), axis=1, fisher=False))
  if(torch.cuda.is_available() and not CPU_MODE): mean, var, kur = mean.cuda(), var.cuda(), kur.cuda()
  return torch.transpose(torch.stack([mean, var, kur]), 1, 0)

def ecdf(data):
  """Compute ECDF for a one-dimensional array of measurements."""
  n = len(data)
  x = np.sort(data)
  y = np.arange(1, n+1) / n
  return x, y

def save_z_cdf(z, real_normal, iterations):
  """saves the cdf plot of the latent space"""
  z_mean = torch.mean(z, dim=0)
  colors = get_cycle_colors()
  fig = plt.figure()
  fig.set_size_inches(8,7) #(16, 12)
  sns.set()
  # add the plot of a true normal sampled vector to compare with the cdf of the z vector
  real_x, real_y = ecdf(real_normal)
  plt.plot(real_x, real_y, next(colors), label="real normal", marker=".", linestyle="none")
  x, y = ecdf(z_mean.cpu().detach().numpy())
  plt.plot(x, y, next(colors), label='z', marker=".", linestyle="none")
  plt.xlabel("x")
  plt.ylabel("CDF")
  plt.legend()
  plt.savefig(SavingPath.get_path(f"_iter-{iterations + 1}.png"), dpi = 100)
  plt.close(fig)

def train_ae(num_epochs, loss_func, ae, dataloader, ae_optimizer, datasetName, searching_mode = False, novel=False):
  print_parameters_ae(num_epochs, loss_func, ae, dataloader, ae_optimizer, datasetName)
  ae_losses = list()
  epoch = 0
  torch.autograd.set_detect_anomaly(True)
  mse_loss = torch.nn.MSELoss()
  factor = 0.005
  z_mse_ = list()
  out_mse_ = list()
  z_regulized_mse_ = list()
  real_normal = torch.randn(10000, dtype=torch.float)

  # checkpoint_interval = RunInterval(hours=0.5)
  checkpoint_interval = RunInterval(minuts=0.5)

  while epoch < num_epochs:
    progress = Progress(len(dataloader), 1)
    # train
    ae.train()
    for batch_count, data in enumerate(dataloader):

      if(checkpoint_interval.try_execute(lambda: save_checkpoint(epoch, ae, ae_optimizer, loss_func, False, "Checkpoints"))):
        print("saved checkpoint")

      if(os.name == "posix"):
        has_input, o, e = select.select( [sys.stdin], [], [], 0.1 ) 
      else:
        has_input = False

      if datasetName.lower() == "mnist": data = data[0]

      if (torch.cuda.is_available() and not CPU_MODE): data = data.cuda()
      ae_optimizer.zero_grad()
      z, outputs = ae(data)
           
      #  in the novel case, compute the loss of the z vector from a normal distribution and add it to the loss of the AE
      if novel:
        z_gauss = gauss_details(z)
        target_gauss = torch.as_tensor([[0.0, 1.0, 3.0]] * z_gauss.shape[0]).type(torch.FloatTensor)
        mse_loss_z = mse_loss(z_gauss.type(torch.FloatTensor), target_gauss)
        z_regulized_mse_.append(factor*mse_loss_z.item())
        loss = loss_func(outputs, data)
        out_mse_.append(loss.item())
        z_mse_.append(mse_loss_z.item())
        batch_loss = loss + factor*mse_loss_z
      else:
        batch_loss = loss_func(outputs.type(torch.FloatTensor), data.type(torch.FloatTensor))
      
      ae_losses.append(batch_loss.cpu().detach().item())
      batch_loss.backward()
      ae_optimizer.step()

      print(f"Epoch:{epoch}/{num_epochs} | progress:{progress.getETA(batch_count)} | loss={batch_loss:.5f}")
      if (searching_mode and progress.isNextPercent(batch_count)):
        save_loss_plot(ae_losses, len(dataloader))
        for i in range(5):
          save_tensor_im(outputs[i], SavingPath.get_path(f"_epoch-{epoch + 1}.batch_count-{batch_count}.{i}.png"))
        if(novel): save_z_cdf(z, real_normal, len(ae_losses))

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
        elif(cli == "plot_z"):
          z_mean = torch.mean(z, dim=0)
          all_zs = z_mean.view(-1)
          all_zs = all_zs.cpu().detach().numpy()
          all_zs.sort()
          plt.plot(range(len(all_zs)), all_zs)
          plt.title(f"Sum={sum(all_zs)}")
          plt.show()            
        else:
          check_cli(cli, ae, epoch, ae_losses, dataloader, ae_optimizer, loss_func)

    if(checkpoint_interval.try_execute(lambda: save_checkpoint(epoch, ae, ae_optimizer, loss_func, True, "Checkpoints"))):
      print("saved checkpoint")
      
    # save plots and images in case we are not in a seaching mode where many are saved inside during a single epoch
    if(not searching_mode): # if we are saving on every iteration, no need to to save the batch
      for i in range(len(outputs)):
        save_tensor_im(outputs[i], SavingPath.get_path(f"_epoch-{epoch+1}.{i}.png"))
    save_loss_plot(ae_losses, len(dataloader))
    
    # Increase the affect of the z error over time.
    if epoch < 10:
      factor *= 2
    epoch += 1

    # free the GPU to avoid memory leak
    del batch_loss
    torch.cuda.empty_cache()

  # show the plot of the different losses
  torch.save(ae.state_dict(), SavingPath.get_path("_ae_param.pkl"))
  plt.plot(np.arange(1, len(z_mse_) + 1), z_mse_)
  plt.plot(np.arange(1, len(out_mse_) + 1), out_mse_)
  plt.plot(np.arange(1, len(z_regulized_mse_) + 1), z_regulized_mse_)
  plt.legend(["z_mse_loss","image_mse_loss", "z_regularized_loss"])
  plt.xlabel("iteration")
  plt.ylabel("batch_loss")
  plt.title("Loss analysis")
  plt.yscale("log")
  plt.show()

# ◄►◄► Prints ◄►◄► #

def print_parameters(num_epochs, loss_func, generator, discriminator, dataloader, im_size, latent_space, unrolled_steps, generator_steps, discriminator_steps, gen_lr, dis_lr,):
  print("◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►")
  print_parameter("num_epochs", num_epochs)
  print_parameter("im_size", im_size)
  print_parameter("latent_space", latent_space)
  print_parameter("unrolled_steps", unrolled_steps)
  print_parameter("generator_steps", generator_steps)
  print_parameter("discriminator_steps", discriminator_steps)
  print_parameter("gen_lr", gen_lr)
  print_parameter("dis_lr", dis_lr)
  print_parameter("batch_size", dataloader.batch_size)
  print_parameter("loss_func", loss_func)
  print_parameter("generator", "M" + re.sub('\D', '', str(generator.model_name)) + ". Trainable params: " + str(count_parameters(generator)))
  print_parameter("discriminator", "M" + re.sub('\D', '', str(discriminator.model_name)) + ". Trainable params: " + str(count_parameters(discriminator)))
  print("◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►")

def print_parameters_ae(num_epochs, loss_func, ae, dataloader, ae_optimizer, datasetName):
  print("◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►")
  print_parameter("num_epochs", num_epochs)
  print_parameter("datasetName", datasetName)
  data_shape = next(iter(dataloader))[0].shape
  print_parameter("data_size", f"{data_shape[0]}x{data_shape[1]}x{data_shape[2]}")
  print_parameter("latent_space", "100")
  print_parameter("ae_optimizer", ae_optimizer)
  print_parameter("batch_size", dataloader.batch_size)
  print_parameter("loss_func", loss_func)
  print("◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►")

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄ #

def gan(args):
  # if in debug mode overwrite the args
  if(args.debug):
    args.epochs = DEBUG_MODE_EPOCHS
    args.batch_size = DEBUG_MODE_BATCH
  # get the losses, datasets and models
  loss_gen, loss_dis = get_loss(args.loss)
  (dataset, im_size) = get_dataset(args.dataset, args.batch_size)
  (gen, dis) = get_models(args, im_size, CPU_MODE)
  # set the path to save the plots and images
  SavingPath(f"{DEBUG_MODE_PATH}_name-{socket.gethostname()}" if args.debug else f"g-{args.gen_model}_d-{args.dis_model}_l-{loss_gen}_e-{args.epochs}_"+
                            f"k-{args.unrolled_steps}_b-{args.batch_size}_z-{args.z_size}_d-{args.dataset}_"+
                            f"gs-{args.generator_steps}_ds-{args.discriminator_steps}_name-{socket.gethostname()}_glr-{args.gen_lr}_dlr-{args.dis_lr}")
  if(args.str != ""): SavingPath.append(f"_{args.str}")                            
  # start training the GAN
  train_gan(args.epochs, loss_gen, loss_dis, gen, dis, dataset, im_size, args.z_size, args.dataset, args.gen_lr, args.dis_lr,
        args.unrolled_steps, args.generator_steps, args.discriminator_steps, args.search_mode)

def ae(args, novel=False):
  # if in debug mode overwrite the args
  if(args.debug):
    args.epochs = DEBUG_MODE_EPOCHS
    args.batch_size = DEBUG_MODE_BATCH
  # get the losses, datasets and models
  _, loss_ae = get_loss(args.loss)
  (dataset, im_size) = get_dataset(args.dataset, args.batch_size)
  ae = get_auto_encoder(args, CPU_MODE)
  # set the path to save the plots and images
  SavingPath(f"{DEBUG_MODE_PATH}_name-{socket.gethostname()}" if args.debug else f"g-{args.gen_model}_d-{args.dis_model}_l-{loss_ae}_e-{args.epochs}_"+
                            f"k-{args.unrolled_steps}_b-{args.batch_size}_lr-{args.gen_lr}_z-{args.z_size}_d-{args.dataset}_"+
                            f"gs-{args.generator_steps}_ds-{args.discriminator_steps}_name-{socket.gethostname()}")

  if(args.str != ""): SavingPath.append(f"_{args.str}")

  ae_optimizer = optim.SGD(ae.parameters(), lr=args.gen_lr, momentum=0.9)

  # save the running configuration to file
  save_info(args, ae, dataset, loss_ae, ae_optimizer, CPU_MODE)

  # start training the AE
  train_ae(args.epochs, loss_ae, ae, dataset, ae_optimizer, args.dataset, args.search_mode, novel)

def main():

  # ◄►◄► Set CLI ◄►◄► #
  parser = argparse.ArgumentParser(description='GAN')
  
  parser.add_argument('-e', '--epochs', type=int, default=DEFAULT_EPOCHS, metavar=('epochs'),
            help=f'Number of epochs to train (default: {DEFAULT_EPOCHS})')
  parser.add_argument('-l', '--loss', type=str, default=DEFAULT_LOSS, metavar=('loss'),
            help=f'Loss function type. Available: mse, non_saturated, cross_entropy. (default: "{DEFAULT_LOSS}")')
  parser.add_argument('-z', '--z-size', type=int, default=DEFAULT_Z, metavar=('latent_space'),
            help=f'Latent space size (default: {DEFAULT_Z})')
  parser.add_argument('-b', '--batch-size', type=int, default=DEFAULT_BATCH, metavar=('batch_size'),
            help=f'Batch size (default: {DEFAULT_BATCH})')
  parser.add_argument('--dataset', default=DEFAULT_DATASET,
            help=f'Sets the dataset type. Availabel: MNIST, celeb (default: "{DEFAULT_DATASET}")')
  parser.add_argument('-g', '--gen-model', type=str, default=DEFAULT_GEN_MODEL, metavar=('model'),
            help=f'Sets the model of the generator. Syntax: Model_x (default: "{DEFAULT_GEN_MODEL}")')
  parser.add_argument('-d', '--dis-model', type=str, default=DEFAULT_DIS_MODEL, metavar=('model'),
            help=f'Sets the model of the discriminator. Syntax: Model_x (default: "{DEFAULT_DIS_MODEL}")')
  parser.add_argument('-k', '--unrolled-steps', type=int, default=DEFAULT_K, metavar=('unrolled_steps'),
            help=f'Sets the unrolling steps of the discriminator. (default: {DEFAULT_K})')
  parser.add_argument('-gs', '--generator-steps', type=int, default=DEFAULT_GENERATOR_STEPS, metavar=('gen_steps'),
            help=f'Sets number of sub-epochs for the generator to. (default: {DEFAULT_GENERATOR_STEPS})')
  parser.add_argument('-ds', '--discriminator-steps', type=int, default=DEFAULT_DISCRIMINATOR_STEPS, metavar=('dis_steps'),
            help=f'Sets number of sub-epochs for the discriminator to. (default: {DEFAULT_DISCRIMINATOR_STEPS})')
  parser.add_argument('--gen_lr', type=float, default=DEFAULT_GEN_LEARNING_RATE, metavar=('rate'),
            help=f'Sets the learning rate of the generators optimizers. (default: {DEFAULT_GEN_LEARNING_RATE})')
  parser.add_argument('--dis_lr', type=float, default=DEFAULT_DIS_LEARNING_RATE, metavar=('rate'),
            help=f'Sets the learning rate of the discriminators optimizers. (default: {DEFAULT_DIS_LEARNING_RATE})')
  parser.add_argument('--debug', action='store_true', default=DEFAULT_DEBUG_MODE,
            help=f'Switches to debug mode which overwrites other given arguments except the loss function (default: {DEFAULT_DEBUG_MODE})')
  parser.add_argument('-sm','--search-mode', action='store_true', default=DEFAULT_SEARCH_MODE,
            help=f'Switches to testing mode which saves the loss function plot frequently (default: {DEFAULT_SEARCH_MODE})')
  parser.add_argument('--release', action='store_true', default=False,
            help=f'Overwrites the debug mode flag (default: {False})')
  parser.add_argument('--str', type=str, default="", metavar=('str'),
            help=f'Appends a string to the end of the path name. (default: "")')                        
  args = parser.parse_args()

  if(args.release): args.debug = False
  
  # initializations
  init_randomness()
  init_flags(CPU_MODE, args.debug)

  # gan(args)
  ae(args, novel=NOVEL)

  SavingPath.finish("-Done")
  # printing the path where all the images and plots where saved
  print(f"Current dir path:\n{SavingPath.get_dir_path()}")

if __name__ == '__main__':
  # change the current directory to the location of the file to read the model and save images and plots in the right location
  os.chdir(os.path.dirname(os.path.realpath(__file__)))
  main()