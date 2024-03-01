# IMPLEMENTATION OF Random Wavelet Convolution
# This data augmentation method is inspired by Progressive Random Convolutions
# https://arxiv.org/pdf/2304.00424.pdf
#
# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

class RandConv_PlusPlus(nn.Module):
    def __init__(self, sigma_max=1., ft_exponent_max=1):
        super(RandConv_PlusPlus, self).__init__()
        self.sigma_max = sigma_max
        self.ft_exponent_max = int(ft_exponent_max)

        self.kernel_size = math.ceil(6 * sigma_max +1)
        if self.kernel_size  % 2 == 0: self.kernel_size  += 1

        self.padding = int((self.kernel_size-1) / 2)
        self.stride = 1

        self.weights = None

    def generate_gaussian(self, out_channels=1, in_channels=1, sigma_min=0.001, sigma_max=0.25, batch_size=1, device=None):
        # Create a 2d coordinate grid
        ax = torch.linspace(-(self.kernel_size - 1) / 2., (self.kernel_size - 1) / 2., self.kernel_size, device=device)
        xx, yy = torch.meshgrid(ax, ax, indexing='ij')
        xx = xx.expand(batch_size, 1, 1, self.kernel_size, self.kernel_size)
        yy = yy.expand(batch_size, 1, 1, self.kernel_size, self.kernel_size)

        # Generate random sigmas for each kernel in the batch, in the range [0.0001, sigma_max+0.0001]
        sigmas = torch.rand(batch_size, out_channels, in_channels, 1, 1, device=device) * (sigma_max-sigma_min) + sigma_min

        # Calculate the 2D Gaussian distribution for all sigmas
        kernels = torch.exp(-0.5 * (xx.pow(2) + yy.pow(2)) / sigmas.pow(2))

        # Normalize each kernel so sum(weights) = 1
        kernels = kernels / ((kernels.sum(dim=[1, 2], keepdim=True))+(1e-7))

        return kernels


    def weight_init(self, batch_size, in_channels, device):

        # Generate random noise sampled from a normal distribution 
        noise = torch.randn(batch_size, in_channels, in_channels, self.kernel_size, self.kernel_size, device=device)

        # Generate batch of 2d Gaussian kernels
        # These are used to constrain the receptive fields
        gaussian_kernels = self.generate_gaussian(sigma_max=self.sigma_max, batch_size=batch_size, in_channels=in_channels, out_channels=in_channels, device=device)

        # Generate a Gaussian window
        # This is a windowing function which reduces spectral leakage
        window = self.generate_gaussian(sigma_max=self.sigma_max, sigma_min=self.sigma_max, device=device)

        # generate exponents in range 1 to ft_exponent_max
        ft_exponents = torch.randint(1, self.ft_exponent_max, (batch_size, in_channels, in_channels, 1, 1), device=device)

        # Apply window to kernel
        windowed_noise = noise * window

        # Take the FFT of the windowed noise
        weights_ft = torch.fft.fft2(windowed_noise)

        # Raise FFT output to the power of ft_exponent, then take the IFT
        weights_ift = torch.real(torch.fft.ifft2(torch.pow(weights_ft, ft_exponents)))

        # Combine Gaussians with the IFT weights
        weighted_kernels = weights_ift * gaussian_kernels

        # Use fobenius norm to normalise kernels to have energy 1
        normalised_kernel = weighted_kernels / (torch.sqrt((weighted_kernels ** 2).sum(dim=(-1, -2, -3), keepdim=True)) + 1e-7)

        self.weights = normalised_kernel


    def __call__(self, x):
        x_size = x.size()
        dim_num = len(x_size)
        if dim_num == 4:
            batch_size, in_channels, height, width = x.size()
        elif dim_num == 5:
            batch_size, num_frames, in_channels, height, width = x.size()

        self.weight_init(batch_size=batch_size, in_channels=in_channels, device=x.device)

        # Measure input image's channel-wise mean and std
        original_mean = x.mean([-1, -2], keepdim=True)
        original_std = x.std([-1, -2], keepdim=True)

        # Reshape input and weights for grouped conv
        if dim_num == 4:
            grouped_x = x.view(1, -1, height, width)
        elif dim_num == 5:
            x = x.permute(1, 0, 2, 3, 4)
            x = x.contiguous()
            grouped_x = x.view(num_frames, -1, height, width)
        #grouped_x = x.view(1, -1, height, width)
        grouped_weights = self.weights.view(-1, in_channels, self.kernel_size, self.kernel_size)

        # Perform convolution, and reshape back to original size
        x = F.conv2d(grouped_x, grouped_weights, stride=self.stride, padding=self.padding, groups=batch_size)

        if dim_num == 5:
            x = x.view(num_frames, batch_size, in_channels, height, width)
            x = x.permute(1, 0, 2, 3, 4)
            x = x.contiguous()
            x = x.view(x_size)
        elif dim_num == 4:
            x = x.view(x_size)
        
        # Standardize the output to have channelwise mean 0 and std 1
        x = (x - x.mean([-2, -1], keepdim=True)) + original_mean
        x = (x / x.std([-2, -1], keepdim=True)) * original_std

        return x

# THIS VERSION OF THE COLLATE FUNCTION DOES NOT USE THE GPU
# THIS IS SO THAT DATA CAN BE PRE-PROCESSED IN THE BACKGROUND DURING TRAINING
# BENCHMARKING IS NECISSARY TO JUSTIFY THIS DECISION

from torch.utils.data.dataloader import default_collate

class RandConv:
    def __init__(self, sigma_max=0.25, ft_exponent_max=10.,  p=0.5):
        self.sigma_max = sigma_max
        self.ft_exponent_max = ft_exponent_max
        self.p = p
        # Create random convolutional layer
        self.RandConv = RandConv_PlusPlus(sigma_max=self.sigma_max, ft_exponent_max=self.ft_exponent_max)

    def __call__(self, batch):
        # Only pass p*100% of the images to the conv layer
        batch_size = len(batch)
        indexes = torch.randperm(batch_size)
        indexes = indexes[:int(self.p * batch_size)]

        # If dataset has images and labels:
        if all(isinstance(sample, tuple) and len(sample)==2 for sample in batch):
            # Run default pytorch collate
            batch = default_collate(batch)
            images, labels = batch
            # Dino augments return lists of images
            if isinstance(images,list):
                for crop in range(len(images)):
                    images[crop][indexes] = self.RandConv(images[crop][indexes])
            else:
                images[indexes] = self.RandConv(images[indexes])
            batch = images, labels
        # If dataset is only images:
        elif all(isinstance(sample, torch.TensorType) for sample in batch):
            # Run default pytorch collate
            batch = default_collate(batch)
            # Run RandConv
            batch[indexes] = self.RandConv(batch[indexes])
        else: raise ValueError("Dataset passed to RandConv_Collate must be either only images or tuples of (image, label)")

        return batch