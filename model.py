# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.ModuleDict):

    def __init__(self, num_inputs, num_outputs, hidden_sizes, model_file_prefix):
        super(Model, self).__init__()

        # MLP section
        layers = []
        ix = 0
        prev_layer = num_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_layer, h))
            layers.append(nn.ReLU(True))
            prev_layer = h

        layers.append(nn.Linear(h, num_outputs))
        self.seq = nn.Sequential(*layers)
        self.model_file = model_file_prefix + "-mlp.pt"
        self.criterion = nn.MSELoss()

    def forward(self, x):
        y_hat = self.seq(x)
        return y_hat

    def loss(self, y, y_hat):
        return self.criterion(y, y_hat)

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    @staticmethod
    def load(file):
        this = Model(model_file=file)
        this.load_state_dict(torch.load(file))
        return this


class VAE(nn.ModuleDict):

    def __init__(self, num_inputs, ZDIMS, ZHID, model_file_prefix ):
        super(VAE, self).__init__()

        # VAE section
        self.encoder = nn.Sequential(
            nn.Linear(num_inputs, ZHID),
            nn.ReLU(True)
        )
        self.encoderMu = nn.Linear(ZHID, ZDIMS)
        self.encoderLogVar = nn.Linear(ZHID, ZDIMS)
        # VAE bottleneck

        self.decoder = nn.Sequential(
            nn.Linear(ZDIMS, ZHID),
            nn.ReLU(True),
            nn.Linear(ZHID, num_inputs),
            nn.Sigmoid()
        )

        self.model_file = model_file_prefix + "-vae.pt"
    def encode(self, x):
        h1 = self.encoder(x)
        mu = self.encoderMu(h1)
        log_var = self.encoderLogVar(h1)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var * 0.5)
        eps = torch.randn_like(std)
        return eps * std + mu

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        # pass through VAE
        mu, log_var = self.encode(x)
        mu2 = self.reparameterize(mu, log_var)
        recon_x = self.decode(mu2)
        return recon_x, mu, log_var

    def loss(self, recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE + KLD

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    @staticmethod
    def load(file):
        this = Model(model_file=file)
        this.load_state_dict(torch.load(file))
        return this
