import sys
sys.path.append('..')

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from skimage.metrics import structural_similarity

from dataset import UDCDataset
from utils import *
from models.restormer import Restormer_fusable
from models.unet import UNet, UNet_fusable
from models.diffusion import DiffUNet


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'


def train_kd(train_dataloader, epochs=200, batch_size=4, distill=False):

    print(f'==> Training for {epochs} Epochs!')
    
    if distill:
        teacher = Restormer_fusable(distill=distill)
        teacher.load_state_dict(torch.load('./PATH_TO_RESTORMER_PRETRAINED.pth', map_location=torch.device(device)))
        teacher = teacher.to(device) 
        teacher.eval()
        print('Loaded Teacher model successfully!')

        model = UNet_fusable()
        print('Loaded Student model successfully!')
        model = model.to(device)
    
    else:
        model = UNet() # Restormer_fusable() if teacher pretraining.
    

    perceptual_loss = VGGLoss()
    
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    psnrs = []

    idx = 0

    pbar = tqdm(total=len(train_dataset) * epochs // batch_size)
    for epoch in range(epochs):
        for sample_noisy, sample_clean in train_dataloader:

            model.train()
            if distill:
                teacher.eval()
            sample_noisy = sample_noisy.to(device)
            sample_clean = sample_clean.to(device)

            if distill: #kd
                f1, f2, f3, f4, f5, f6, f7, teacher_output = teacher(sample_noisy)
                # denoise
                denoised_sample = model(sample_noisy, f1, f2, f3, f4, f5, f6, f7)
            else:
                denoised_sample = model(sample_noisy)

            # kd loss combined
            l2_gt = torch.mean((denoised_sample - sample_clean)**2)
            l2_kd = torch.mean((denoised_sample - teacher_output)**2) if distill else 0

            perceptual_loss_gt = perceptual_loss(denoised_sample, sample_clean)
            perceptual_loss_kd = perceptual_loss(denoised_sample, teacher_output) if distill else 0

            loss = l2_gt + l2_kd + perceptual_loss_gt + perceptual_loss_kd

            psnr = calc_psnr(denoised_sample, sample_clean)

            losses.append(loss.item())
            psnrs.append(psnr)

            # update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            idx += 1
            pbar.update(1)

        if epoch%10==0:
            print('Epoch: {}, PSNR: {}'.format(epoch, np.mean(psnrs)))
    pbar.close()
    return model, np.mean(psnrs)


def train_diffusion(train_dataloader, model_params, epochs=100, batch_size=1):

    print(f'==> Training for {epochs} Epochs!')
    
    model = DiffUNet(**model_params)
    model = model.to(device)

    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    losses = []
    psnrs = []

    idx = 0

    pbar = tqdm(total=len(train_dataset) * epochs // batch_size)
    for epoch in range(epochs):
        for sample_noisy, sample_clean in train_dataloader:
            # Send to device
            sample_noisy, sample_clean = sample_noisy.to(device), sample_clean.to(device)

            # Put in training mode
            model.train()
            
            # Get model output
            denoised_sample = model(sample_noisy)

            # Compute L2 Loss
            l2_gt = torch.mean((denoised_sample - sample_clean)**2)

            loss = l2_gt

            # Compute PSNR
            psnr = calc_psnr(denoised_sample, sample_clean)

            losses.append(loss.item())
            psnrs.append(psnr)

            # Update model
            loss.backward()
            optim.step()
            optim.zero_grad()

            idx += 1
            pbar.update(1)

        if epoch%10==0:
            print('Epoch: {}, PSNR: {}'.format(epoch, np.mean(psnrs)))
    pbar.close()
    return model, np.mean(psnrs)


if __name__ == "__main__":
    dataset_type = 'Toled' # or 'Poled'
    train_dataset = UDCDataset(split=dataset_type, eval=False)
    print(f'=> {dataset_type} Dataset created!')
    
    EXPERIMENT = 'DIFFUSION'  # One of {'KD', 'DIFFUSION'}

    if EXPERIMENT == 'KD':
        # TRAINING KNOWLEDGE DISTILLATION MODELS
        BATCH_SIZE = 4
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        EPOCHS = 200
        DISTILL = False
        trained_model, train_psnr = train_kd(train_dataloader, epochs=EPOCHS, distill=DISTILL)

        # torch.save(trained_model.state_dict(), './MODEL_NAME.pth')

        print("Finished Training!")
        print('Training PSNR: {}'.format(train_psnr))

    elif EXPERIMENT == 'DIFFUSION':
        # TRAINING DIFFUSION MODELS
        BATCH_SIZE = 1          
        train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        EPOCHS = 100
        MODEL_CONFIG = {'in_channels': 3, 
                        'out_channels': 3, 
                        'hidden_channels': 32, 
                        'kernel_size': 3,
                        'hidden_layers': 15, 
                        'use_bias': True, 
                        'steps': 1, 
                        'large_model': False}
        trained_model, train_psnr = train_diffusion(train_dataloader, epochs=EPOCHS, model_params=MODEL_CONFIG)

        # torch.save(trained_model.state_dict(), './MODEL_NAME.pth')

        print("Finished Training!")
        print('Training PSNR: {}'.format(train_psnr))
