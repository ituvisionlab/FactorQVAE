import numpy as np
import os
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms

import pytorch_lightning as pl

def shift(x):
    return x - 0.5

class MPI3DComplex(Dataset):
    def __init__(self, config):
        self.root = config.data.data_path
        self.img_paths = os.listdir(self.root)
        self.img_size = 64
        self.n_samples = len(self.img_paths)
        self.factor_sizes = [4, 4, 2, 3, 3, 40, 40] # object_color, object_shape, object_size, camera_height, background_color, horizontal_axis, vertical_axis
        self.latent_factor_indices = list(range(7))
        self.num_total_factors = 7
        self.factor_bases = np.prod(self.factor_sizes) / np.cumprod(self.factor_sizes) # array([115200, 28800, 14400, 4800, 1600, 40, 1])
        self.observation_factor_indices = [i for i in range(self.num_total_factors) if i not in self.latent_factor_indices]
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(shift)])
    
    def __len__(self):
        return self.n_samples
    
    def __getitem__(self, index):
        img_path = "{}/{}.png".format(self.root, index)        
        img = Image.open(img_path)
        img = self.transform(img)

        return img

class MPI3DComplexDataModule(pl.LightningDataModule):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.seed = config.train.seed
    
    def setup(self, stage=None):
        self.dataset = MPI3DComplex(self.config)
        self.n_samples = len(self.dataset)
        self.n_train = int(self.n_samples * 0.8)
        self.n_val = int(self.n_samples * 0.1)
        self.n_test = self.n_samples - self.n_train - self.n_val
        generator = torch.Generator().manual_seed(self.seed)
        self.train, self.val, self.test = random_split(self.dataset, [self.n_train, self.n_val, self.n_test], generator=generator)
        
    def train_dataloader(self):
        return DataLoader(self.train, 
                        batch_size=self.config.data.batch_size, 
                        shuffle=True, 
                        num_workers=self.config.data.num_workers,
                        pin_memory=True)
    
    def val_dataloader(self):
        return DataLoader(self.val,
                        batch_size=self.config.data.batch_size, 
                        shuffle=False, 
                        num_workers=self.config.data.num_workers,
                        pin_memory=True)
    
    def test_dataloader(self):
        return DataLoader(self.test,
                        batch_size=self.config.data.batch_size, 
                        shuffle=False, 
                        num_workers=self.config.data.num_workers,
                        pin_memory=True)

