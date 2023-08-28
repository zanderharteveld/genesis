# coding: utf-8
# standard libraries
import os
import sys
import glob
import json
import time
import numpy as np
import pandas as pd

# nn libraries
import torch
from torch import nn

# this library
from .utils import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

################################################################################
################################ VAE definition ################################
################################################################################


class Encoder1(nn.Module):
    """A simple implementation of a convolutional Encoder."""

    def __init__(self, image_channels, image_shape, hidden, latent):
        super(Encoder1, self).__init__()

        self.hidden = hidden
        self.latent = latent

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=32, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ELU(),
            nn.Conv2d(32, 64, kernel_size=16, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ELU(),
            nn.Conv2d(64, 128, kernel_size=8, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ELU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ELU(),
        )

        # fully connected layers for learning representations
        self.lin_mean = nn.Linear(self.hidden, self.latent)
        self.lin_var = nn.Linear(self.hidden, self.latent)
        self.training = True

    def forward(self, x):
        batch = x.shape[0]
        x = self.encoder(x)
        h_ = x.view(batch, -1)
        mean = self.lin_mean(h_)
        log_var = self.lin_var(h_)  # encoder produces mean and log of variance
        # (i.e., parateters of simple tractable
        #        normal distribution "q")
        var = torch.exp(0.5 * log_var)  # takes exponential function

        z = self.reparameterization(mean, var)
        z = z.view(batch, self.latent, 1, 1)
        return z, mean, log_var

    def reparameterization(
        self,
        mean,
        var,
    ):
        epsilon = torch.rand_like(var).to(device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z


class Decoder1(nn.Module):
    """A simple implementation of a Decoder."""

    def __init__(self, image_channels, image_shape, hidden, latent):
        super(Decoder1, self).__init__()

        self.image_shape = image_shape
        self.latent = latent

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, 256, kernel_size=8, stride=4),
            nn.InstanceNorm2d(256, affine=True),
            nn.ELU(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.InstanceNorm2d(128, affine=True),
            nn.ELU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.InstanceNorm2d(64, affine=True),
            nn.ELU(),
        )

        self.dist_probs = nn.Sequential(
            nn.Conv2d(64, 37, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(37, affine=True),
            nn.Softmax(dim=1),
        )

        self.omega_probs = nn.Sequential(
            nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(25, affine=True),
            nn.Softmax(dim=1),
        )

        self.theta_probs = nn.Sequential(
            nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(25, affine=True),
            nn.Softmax(dim=1),
        )

        self.phi_probs = nn.Sequential(
            nn.Conv2d(64, 13, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(13, affine=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x_hat = self.decoder(x)
        # to probs
        x_hat_dist = self.dist_probs(x_hat).permute(0, 2, 3, 1)
        x_hat_dist = 0.5 * (x_hat_dist + x_hat_dist.permute((0, 2, 1, 3)))  # symmetric
        x_hat_omega = self.omega_probs(x_hat).permute(0, 2, 3, 1)
        x_hat_omega = 0.5 * (
            x_hat_omega + x_hat_omega.permute((0, 2, 1, 3))
        )  # symmetric
        x_hat_theta = self.theta_probs(x_hat).permute(0, 2, 3, 1)  # non symmetric
        x_hat_phi = self.phi_probs(x_hat).permute(0, 2, 3, 1)  # non symmetric
        return [x_hat_dist, x_hat_omega, x_hat_theta, x_hat_phi]


class Model1(nn.Module):
    """A simple implementation of a VAE."""

    def __init__(self, Encoder, Decoder):
        super(Model1, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var


class Encoder2(nn.Module):
    """A simple implementation of a convolutional Encoder."""

    def __init__(self, image_channels, image_shape, hidden, latent):
        super(Encoder2, self).__init__()

        self.hidden = hidden
        self.latent = latent

        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=32, stride=2, padding=1),
            nn.InstanceNorm2d(32, affine=True),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Conv2d(32, 64, kernel_size=16, stride=2, padding=1),
            nn.InstanceNorm2d(64, affine=True),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Conv2d(64, 128, kernel_size=8, stride=2, padding=1),
            nn.InstanceNorm2d(128, affine=True),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(256, affine=True),
            nn.ELU(),
        )

        # fully connected layers for learning representations
        self.lin_mean = nn.Linear(self.hidden, self.latent)
        self.lin_var = nn.Linear(self.hidden, self.latent)
        self.training = True

    def forward(self, x):
        batch = x.shape[0]
        x = self.encoder(x)
        h_ = x.view(batch, -1)
        mean = self.lin_mean(h_)
        log_var = self.lin_var(h_)  # encoder produces mean and log of variance
        # (i.e., parateters of simple tractable
        #        normal distribution "q")
        var = torch.exp(0.5 * log_var)  # takes exponential function

        z = self.reparameterization(mean, var)
        z = z.view(batch, self.latent, 1, 1)
        return z, mean, log_var

    def reparameterization(
        self,
        mean,
        var,
    ):
        epsilon = torch.rand_like(var).to(device)  # sampling epsilon
        z = mean + var * epsilon  # reparameterization trick
        return z


class Decoder2(nn.Module):
    """A simple implementation of a Decoder."""

    def __init__(self, image_channels, image_shape, hidden, latent):
        super(Decoder2, self).__init__()

        self.image_shape = image_shape
        self.latent = latent

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(latent, 256, kernel_size=8, stride=4),
            nn.InstanceNorm2d(256, affine=True),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=4),
            nn.InstanceNorm2d(128, affine=True),
            nn.ELU(),
            nn.Dropout(0.4),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=4),
            nn.InstanceNorm2d(64, affine=True),
            nn.ELU(),
        )

        self.dist_probs = nn.Sequential(
            nn.Conv2d(64, 37, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(37, affine=True),
            nn.Softmax(dim=1),
        )

        self.omega_probs = nn.Sequential(
            nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(25, affine=True),
            nn.Softmax(dim=1),
        )

        self.theta_probs = nn.Sequential(
            nn.Conv2d(64, 25, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(25, affine=True),
            nn.Softmax(dim=1),
        )

        self.phi_probs = nn.Sequential(
            nn.Conv2d(64, 13, kernel_size=1, stride=1, padding=0),
            nn.InstanceNorm2d(13, affine=True),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        x_hat = self.decoder(x)
        # to probs
        x_hat_dist = self.dist_probs(x_hat).permute(0, 2, 3, 1)
        x_hat_dist = 0.5 * (x_hat_dist + x_hat_dist.permute((0, 2, 1, 3)))  # symmetric
        x_hat_omega = self.omega_probs(x_hat).permute(0, 2, 3, 1)
        x_hat_omega = 0.5 * (
            x_hat_omega + x_hat_omega.permute((0, 2, 1, 3))
        )  # symmetric
        x_hat_theta = self.theta_probs(x_hat).permute(0, 2, 3, 1)  # non symmetric
        x_hat_phi = self.phi_probs(x_hat).permute(0, 2, 3, 1)  # non symmetric
        return [x_hat_dist, x_hat_omega, x_hat_theta, x_hat_phi]


class Model2(nn.Module):
    """A simple implementation of a VAE."""

    def __init__(self, Encoder, Decoder):
        super(Model2, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder

    def forward(self, x):
        z, mean, log_var = self.Encoder(x)
        x_hat = self.Decoder(z)

        return x_hat, mean, log_var
