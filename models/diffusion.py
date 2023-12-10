import deepinv as dinv
import torch
import torch.nn as nn
import numpy as np

########## PRETRAINED DIFFUSION U-NET WITH DnCNN NOISE ESTIMATOR
class DiffUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, hidden_channels=32, kernel_size=3,
                 hidden_layers=15, use_bias=True, steps=1, large_model=False):
        super(DiffUNet, self).__init__()

        # Load pretrained Diffusion U-Net from DeepInverse library
        if large_model:
            self.model = dinv.models.DiffUNet(large_model=True)
        else:
            self.model = dinv.models.DiffUNet(pretrained='../models/diffusion_ffhq_10m.pt')
        
        # Original model code is hardcoded for running this function on GPU, so change it to CPU
        self.model.find_nearest = self.find_nearest_fixed

        # Noise estimator DnCNN
        layers = []
        layers.append(torch.nn.Conv2d(in_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
        layers.append(torch.nn.ReLU(inplace=True))
        for i in range(hidden_layers):
            layers.append(torch.nn.Conv2d(hidden_channels, hidden_channels, kernel_size, padding='same', bias=use_bias))
            layers.append(torch.nn.ReLU(inplace=True))
        layers.append(torch.nn.Conv2d(hidden_channels, out_channels, kernel_size, padding='same', bias=use_bias))
        self.noise_est_base = nn.Sequential(*layers)

        # Final dense layer for converting to scalar noise variance estimate
        self.noise_final = nn.Linear(out_channels, 1)
        
        # No. of times to iteratively denoise
        self.steps = steps

    def forward(self, x):
        out = x

        # Iteratively for each step from 1 to total steps specified
        for step in range(self.steps):
            # Get feature map from convolutional layers of DnCNN
            noise_rep = self.noise_est_base(out)
            # Perform GAP over spatial dimensions
            noise_rep = torch.mean(noise_rep.view(noise_rep.size(0), noise_rep.size(1), -1), dim=2)
            # Pass through dense layer to get scalar noise variance estimate
            noise_est = self.noise_final(noise_rep)

            # Pass estimate and input image into pretrained diffusion U-Net
            out = self.model(out, noise_est, type_t='noise_level')

        return out

    def find_nearest_fixed(self, array, value):
        """
        Find the argmin of the nearest value in an array.
        """
        array = np.asarray(array)
        if isinstance(value, torch.Tensor):
            value = value.cpu().detach().numpy()  # Changed this with .cpu()
        idx = (np.abs(array - value)).argmin()
        return idx
    