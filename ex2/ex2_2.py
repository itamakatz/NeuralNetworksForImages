import sys
import socket
import numpy as np
import argparse
import matplotlib.pyplot as plt
import os
from os import listdir
from os.path import isfile, join
from utils import SavingPath, Running_Time, Progress, save_im, get_cycle_colors
from common import get_loss, get_dataset, get_models, get_auto_encoder
if(socket.gethostname() == 'urim'):
    save_path = sys.path
    sys.path.clear()
    sys.path.append('/usr/local/lib/python3.6/dist-packages')
import torch
import torch.optim as optim
if(socket.gethostname() == 'urim'):
    sys.path  = save_path
    sys.path.append('/usr/local/lib/python3.6/dist-packages')

# ◄►◄►◄►◄►◄► Magic Variables ◄►◄►◄►◄►◄► #

# DEFAULT_NAME = "g-M1_d-M3_l-BCEWithLogitsLoss()_e-20_k-6_b-24_z-50_d-MNIST_generator_param.pkl"
# DEFAULT_NAME = "g-M1_d-M1_l-MSELoss()_e-100_k-0_b-24_z-100_d-MNIST_gs-1_ds-1_name-itamar-ubuntu_auto_encoder_param.pkl"
DEFAULT_NAME = "g-M1_d-M1_l-MSELoss()_e-50_k-0_b-48_z-500_d-MNIST_gs-1_ds-1_name-itamar-ubuntu_auto_encoder_param.pkl"

DEFAULT_IGAN_ITERATIONS = 20000 # "i" is for inverse
DEFAULT_STARTING_COUNT = 5
# DEFAULT_DEBUG_MODE = True
DEFAULT_DEBUG_MODE = False

DEBUG_MODE_PATH = "Debugging"
ENCODER_MODEL = "M1"
DECODER_MODEL = "M1"
# CPU_MODE = True
CPU_MODE = False

# CUDA_LAUNCH_BLOCKING=1 # flag to debug pytorch

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►#

def train_inverse_gan(gen, iterations, z_size, target_im, num_of_zs):
    
    # initialization
    if (torch.cuda.is_available() and not CPU_MODE): target_im = target_im.cuda()
    z_list = list()
    losses_list = list()
    optimizers = list()
    # getting the mse loss function
    progress = Progress(iterations)
    loss, _ = get_loss("mse")

    # run the inversion for num_of_zs times and return the best resulting z
    for i in range(num_of_zs):
        z_list.append(torch.FloatTensor(1, z_size).normal_(0, 1))
        if (torch.cuda.is_available() and not CPU_MODE): z_list[i] = z_list[i].cuda()
        z_list[i].requires_grad = True
        losses_list.append(list())
        optimizers.append(optim.SGD([z_list[i]], lr=0.001, momentum=0.9))
    
    # for each z, we iterate over iterations times where in each iteration we optimize over the z vector itself 
    # given the loss of the generated image with the input image.
    for iter in range(iterations):
        for i in range(num_of_zs):
            optimizers[i].zero_grad()
            gen_z = gen(z_list[i])
            loss_g_z = loss(gen_z, target_im)
            losses_list[i].append(loss_g_z.item())
            loss_g_z.backward()
            optimizers[i].step()
        progress.printProgress(iter)
    
    # getting the index of the best scoring z vector to be returned
    best_opt = torch.argmin(torch.FloatTensor([losses_list[i][iterations - 1] for i in range(num_of_zs)]))
    # given the best z vector, generate its corresponding image from the generator
    final_gen_z = gen(z_list[best_opt])
    # saving the target and inversed images 
    save_im(target_im.cpu().numpy().reshape((target_im.shape[1],target_im.shape[2],target_im.shape[3])),
            SavingPath.get_path(f"target_im.png"))
    save_im(final_gen_z.cpu().detach().numpy().reshape((final_gen_z.shape[1],final_gen_z.shape[2],final_gen_z.shape[3])),
            SavingPath.get_path(f"inversed_im.png"))

    # save the loss functions of all z's as a function of iteration in a single plot
    colors = get_cycle_colors()
    fig = plt.figure()
    for i in range(len(losses_list)):
        plt.plot(np.arange(1, iterations + 1), losses_list[i], next(colors))
    plt.xlabel("Iteration")
    plt.ylabel("Mse Loss")
    plt.savefig(SavingPath.get_path(f"_all_z_losses.png"))
    plt.close(fig)

    return z_list[best_opt], losses_list[best_opt]

# ◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄►◄► #

def inverse_gan(args, PATH, real_im = True, mask="none"):
    
    # initializing/preparing data
    SavingPath(DEBUG_MODE_PATH if args.debug_mode else f"Inverse_Gan_z-{args.z_size}_i-{args.iterations}_s-{args.starting_count}_dataset-{args.dataset}")
    iterations = args.iterations
    dataset, im_size = get_dataset(args.dataset, args.batch_size)
    gen_model, _ = get_models(args, im_size, CPU_MODE)
    gen_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    
    for param in gen_model.parameters():
        param.requires_grad = False
    gen_model.eval()
    # randomly picking images from the dataset for finding its corresponding latent vector z
    target_im = get_target_im(dataset, gen_model, real_im)
    masked_target_im = apply_mask(target_im, mask)
    #  training and searching for the best latent vector z
    z, losses = train_inverse_gan(gen_model, iterations, args.z_size, masked_target_im, args.starting_count)
    SavingPath.finish("-Done")

def apply_mask(img, type):
    '''creates a mask out of 3 possible mask for the image'''
    # initializing
    mn = img.min()
    im_shape = img.shape[2]
    img_clone = img.clone()

    # depending on the type of the mask return the corresponding mask
    if type.lower() == "pepper":
        row_ind = np.random.randint(28, size=int(im_shape * im_shape * 0.3))
        col_ind = np.random.randint(28, size=int(im_shape * im_shape * 0.3))
        img_clone[:,:,row_ind,col_ind] = mn
        return img_clone
    if type.lower() == "square":
        top_left_corner_row = np.random.randint(im_shape - 8, size=1)
        top_left_corner_col = np.random.randint(im_shape - 8, size=1)
        square_grid = np.meshgrid(np.arange(top_left_corner_row, top_left_corner_row + 8),
                           np.arange(top_left_corner_col, top_left_corner_col + 8))
        img_clone[:,:,square_grid] = mn
        return img_clone
    if type.lower() == "crosswalk":
        img_clone[:,:,::2] = mn
        return img_clone
    if type.lower() == "none":
        return img_clone

def ae_novel_tester(PATH, ae_model, z_size, num_of_zs):
    """test the AE in the case we forced the latent space to be a normal distribution.
    run with encoder model M1, decoder model M1"""

    SavingPath("auto_encoder_novel_images_tester")
    ae_model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
    z_batch = torch.randn((1, num_of_zs, z_size), dtype=torch.float)
    z_batch = torch.mean(z_batch, dim=0)
    z_batch[-1] = torch.randn((z_size), dtype=torch.float)
    if(torch.cuda.is_available() and not CPU_MODE): z_batch = z_batch.cuda()
    predictions = ae_model.decoder(z_batch)
    for i in range(len(predictions)):
        save_im(predictions[i].cpu().detach().numpy().reshape(
            (predictions.shape[1], predictions.shape[2], predictions.shape[3])),
                SavingPath.get_path(f"_pred_{i}.png"))
    SavingPath.finish("-Done")

def interpolate_gan(args, path):
    '''generating images using the GAN model by means of interpolating two distinct latent vectors 
    which in turn generate a new interpolated image'''

    SavingPath(DEBUG_MODE_PATH if args.debug_mode else f"Inverse_Gan_z-{args.z_size}_i-{args.iterations}_s-{args.starting_count}_dataset-{args.dataset}")
    dataset, im_size = get_dataset(args.dataset, args.batch_size)
    im_1 = torch.unsqueeze(next(iter(dataset))[0][0], 0)
    im_2 = torch.unsqueeze(next(iter(dataset))[0][0], 0)
    print(im_1.shape)
    gen_model, _ = get_models(args, im_size, CPU_MODE)
    gen_model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    for param in gen_model.parameters():
        param.requires_grad = False
    gen_model.eval()
    SavingPath(f"GAN_interpolation")
    z1, losses = train_inverse_gan(gen_model, args.iterations, args.z_size, im_1, args.starting_count)
    z2, losses = train_inverse_gan(gen_model, args.iterations, args.z_size, im_2, args.starting_count)
    count = 1
    for alpha in [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]:
        z3 = z1 * alpha + z2 * (1 - alpha)
        output3 = gen_model(z3)
        save_im(output3.cpu().detach().numpy().reshape(
            (output3.shape[1], output3.shape[2], output3.shape[3])),
            SavingPath.get_path(f"_{count}_alpha-{alpha}.png"))
        count += 1
    SavingPath.finish("-Done")

def interpolate_AE(auto_encoder, auto_encoder_path):
    '''generating images using the AE model by means of interpolating two distinct latent vectors 
    which in turn generate a new interpolated image'''    
    dataset, im_size = get_dataset("mnist", 10)
    im_1 = torch.unsqueeze(next(iter(dataset))[0][0], 0)
    im_2 = torch.unsqueeze(next(iter(dataset))[0][0], 0)
    if (torch.cuda.is_available() and not CPU_MODE): im_1, im_2 = im_1.cuda(), im_2.cuda()
    auto_encoder.load_state_dict(torch.load(auto_encoder_path, map_location=torch.device('cpu')))
    for param in auto_encoder.parameters():
        param.requires_grad = False
    auto_encoder.eval()
    SavingPath(f"interpolate_AE")
    z1, output1 = auto_encoder(im_1)
    z2, output2 = auto_encoder(im_2)
    count = 1
    for alpha in [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1]:
        z3 = z1 * alpha + z2 * (1 - alpha)
        output3 = auto_encoder.decoder(z3)
        save_im(output3.cpu().detach().numpy().reshape(
            (output3.shape[1], output3.shape[2], output3.shape[3])),
            SavingPath.get_path(f"_{count}_alpha-{alpha}_interpolate_AE.png"))
        count += 1

def get_path():
    current_path = os.path.dirname(os.path.realpath(__file__))
    onlyfiles = [f for f in listdir(current_path) if isfile(join(current_path, f)) and f.endswith(".pkl")]
    if(len(onlyfiles) != 1):
        raise Exception(f"There is either no .pkl file in {current_path} or there are too many")
    return onlyfiles[0]

def get_target_im(dataset, gen, real_img = True):
    if real_img:
        return torch.unsqueeze(next(iter(dataset))[0][0], 0)
    else:
        z_tag = torch.randn((1, 50), dtype=torch.float)
        if(torch.cuda.is_available() and not CPU_MODE):
            z_tag = z_tag.cuda()
        return gen(z_tag)

def read_args_from_file(path):
    z = [int(s) for s in (path.split("z-")[1].split("_")) if s.isdigit()][0]
    batch_size = [int(s) for s in (path.split("b-")[1].split("_")) if s.isdigit()][0]
    gen_model = [s for s in (path.split("g-")[1].split("_"))][0]
    dis_model = [s for s in (path.split("d-")[1].split("_"))][0]
    dataset = "MNIST" if "MNIST" in path or "MINST" in path else "celebs"

    return {"z": z, "batch_size": batch_size, "gen_model": gen_model, "dis_model": dis_model, "dataset": dataset, "path": path}

def main():
    Running_Time.init()
    # ◄►◄► Set CUDA GPU ◄►◄► #
    if(torch.cuda.is_available() and not CPU_MODE):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        torch.cuda.set_device(device)
        print("is CUDA initialized: "+ "YES" if torch.cuda.is_initialized() else "NO")

    path = DEFAULT_NAME if(DEFAULT_NAME) else get_path()
    file_args = read_args_from_file(path)

    parser = argparse.ArgumentParser(description='IGAN')

    parser.add_argument('-z', '--z-size', type=int, default=file_args["z"], metavar=('latent_space'),
                        help=f'Latent space size (default: {file_args["z"]})')
    parser.add_argument('-b', '--batch-size', type=int, default=file_args["batch_size"], metavar=('batch_size'),
                        help=f'Batch size (default: {file_args["batch_size"]})')
    parser.add_argument('--dataset', default=file_args["dataset"],
                        help=f'Sets the dataset type. Availabel: MNIST, celeb (default: "{file_args["dataset"]}")')
    parser.add_argument('-g', '--gen-model', type=str, default=file_args["gen_model"], metavar=('model'),
                        help=f'Sets the model of the generator. Syntax: Model_x (default: "{file_args["gen_model"]}")')
    parser.add_argument('-d', '--dis-model', type=str, default=file_args["dis_model"], metavar=('model'),
                        help=f'Sets the model of the discriminator. Syntax: Model_x (default: "{file_args["dis_model"]}")')

    parser.add_argument('-i', '--iterations', type=int, default=DEFAULT_IGAN_ITERATIONS, metavar=('iterations'),
                        help=f'Number of iterations for converging the search of the latent vector z (default: {DEFAULT_IGAN_ITERATIONS})')
    parser.add_argument('-s', '--starting-count', type=int, default=DEFAULT_STARTING_COUNT, metavar=('count'),
                        help=f'Number of different random starting vectors z when searching the best overall (default: {DEFAULT_STARTING_COUNT})')
    parser.add_argument('--debug-mode', action='store_true', default=DEFAULT_DEBUG_MODE,
                        help=f'Switches to debug mode which overwrites other given arguments except the loss function (default: {DEFAULT_DEBUG_MODE})')
    args = parser.parse_args()

    """ Inverse GAN (generative model Q2, Q3)"""
    # inverse_gan(args, file_args["path"], real_im = True, mask="none")
    """ GAN interpolation """
    # interpolate_gan(args, file_args["path"])
    """ auto-encoder novel images results (Non adversial models Q2)"""
    ae_novel_tester(DEFAULT_NAME, get_auto_encoder(args, CPU_MODE), args.z_size, 600)
    """ AE interpolation """
    # interpolate_AE(get_auto_encoder(args, CPU_MODE), path)
main()