import torch
from torch.utils.data import Dataset
from glob import glob
import os
import numpy as np
import skimage.io
from skimage.transform import resize

class UDCDataset(Dataset):
    def __init__(self, root='../../udc_data_2020/', split='Toled',eval=False):
        self.eval = eval
        # Determine whether train or test images need to be loaded
        if not self.eval:
            self.files_noisy = sorted(glob(os.path.join(root, 'UDC_train', split, 'LQ','*')))
            self.files_clean = sorted(glob(os.path.join(root, 'UDC_train', split, 'HQ','*')))
        else:
            self.files_noisy = sorted(glob(os.path.join(root, 'UDC_val_test', split, 'LQ','*')))
            self.files_clean = sorted(glob(os.path.join(root, 'UDC_val_test', split, 'HQ','*')))

        self.images_noisy = self.load_images(self.files_noisy)
        self.images_clean = self.load_images(self.files_clean)

    def load_images(self, files):
        out = []
        # For each image file
        for fname in files:
            # Load in image
            img = skimage.io.imread(fname)
            
            # Resize to 256 x 512 for easier computation, anti_aliasing to remove artifacts
            # This gives image in range [0, 1] already
            img = resize(img, (256, 512), anti_aliasing=True)

            # Ensure the lower size dimension is first
            if img.shape[0] > img.shape[1]:
                img = img.transpose(1, 0, 2)
            # Ensure the channels dimension is first
            img = img.transpose(2, 0, 1).astype(np.float32) #/255
            out.append(torch.from_numpy(img))

        # Stack examples
        return torch.stack(out)

    def __len__(self):
        return self.images_noisy.shape[0]

    def __getitem__(self, idx):
        return self.images_noisy[idx], self.images_clean[idx]