import sys
sys.path.append('..')


import numpy as np
import torch
from skimage.metrics import structural_similarity
from time import time

from dataset import UDCDataset
from utils import *
from models.restormer import Restormer_fusable
from models.unet import UNet, UNet_fusable
from models.diffusion import DiffUNet

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

def evaluate_model(model,split):
    
    # Get validation dataset
    dataset = UDCDataset(split=split, eval=True)
    model.eval()

    psnrs = []
    ssims = []
    times = []
    for idx,images in enumerate(dataset):
        # Get noisy and GT samples from the dataset
        image_noisy, image_clean = images[0], images[1]

        # Add batch dimension
        image_noisy = image_noisy[None, ...].to(device)  
        image_clean = image_clean.to(device)

        # Time the image throughput through the model
        start = time()
        
        # Get model output
        denoised_image = model(image_noisy)
        end = time()
        times.append(end-start)

        # Get numpy arrays
        denoised_image = denoised_image.squeeze().detach().cpu().permute(1,2,0).numpy()
        image_clean = image_clean.detach().cpu().permute(1,2,0).numpy()

        # Compute PSNR and SSIM metrics and append to list
        psnr = calc_psnr(denoised_image, image_clean)
        ssim = structural_similarity(denoised_image, image_clean, channel_axis=-1, data_range=1)
        psnrs.append(psnr)
        ssims.append(ssim)


        # Save sample images
        # if idx == 3:
            # image_noisy = (image_noisy.squeeze().detach().cpu().permute(1,2,0).numpy()*255.).astype(np.uint8)
            # denoised_image = (denoised_image.squeeze().detach().cpu().permute(1,2,0).numpy()*255.).astype(np.uint8)
            # image_clean = (image_clean.squeeze().detach().cpu().permute(1,2,0).numpy()*255.).astype(np.uint8)
            # print(denoised_image.shape, image_clean.shape)
            # skimage.io.imsave('./random_tests/results/input_noisy_FULL_{}_{}.png'.format(split, idx), image_noisy)
            # skimage.io.imsave('./random_tests/results/input_clean_FULL_{}_{}.png'.format(split, idx), image_clean)
            # skimage.io.imsave('./random_tests/results/output_FULL_{}_{}.png'.format(split, idx), denoised_image)

    # Return averages of the metrics
    print('Time Per Image: {}'.format(round(np.mean(times), 5)))
    return round(np.mean(psnrs),2), round(np.mean(ssims),2)

if __name__ == "__main__":

    dataset_type = 'Toled' # or 'Poled'

    # Specify model type here
    trained_model = UNet_fusable()

    # Load in trained weights
    trained_model.load_state_dict(torch.load('./PATH_TO_TRAINED_MODEL.pth', map_location=device))
    print('Loaded Weights successfully!')
    trained_model = trained_model.to(device) 

    # Get PSNR and SSIM metrics
    p, s = evaluate_model(trained_model, split=dataset_type)
    print('Eval PSNR: {}, Eval SSIM: {}'.format(p, s))