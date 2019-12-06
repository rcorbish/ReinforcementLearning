# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model :
    def __init__(self, num_inputs, num_outputs, hidden_sizes, model_file_prefix):
        self.mlp = MLP( num_inputs, num_outputs, hidden_sizes, model_file_prefix )
        self.vae = MLP( num_inputs, num_outputs, hidden_sizes, model_file_prefix )

    def forward(self, x):
        y_hat = self.mlp(x)
        return y_hat

    def loss(self, y, y_hat):
        return self.mlp.loss( y, y_hat ) 

    def save(self):
        '''
        Write parameters to disk
        '''
        self.mlp.save()
        self.vae.save()

    @staticmethod
    def load(file):
        pass



class MLP(nn.ModuleDict):
    '''
    This learns the value of a sequence of observations. 
    It's a custom hidden layer sized multi-layer peceptron
    '''
    def __init__(self, num_inputs, num_outputs, hidden_sizes, model_file_prefix):
        '''
        num_inputs - number of input values
        num_outputs - should be 1 for the single value
        hidden_sizes - list of sizes of hidden layers
        model_file_prefix - for saving paramters. "-vae.pt" added for actual name
        '''
        super(MLP, self).__init__()

        layers = []     # list of layers starting at input layer
        ix = 0
        prev_layer = num_inputs
        for h in hidden_sizes:
            layers.append(nn.Linear(prev_layer, h))
            layers.append(nn.LeakyReLU(True))
            prev_layer = h

        layers.append(nn.Linear(h, num_outputs))    # add final output layer

        self.seq = nn.Sequential(*layers)           # construct the model
        self.model_file = model_file_prefix + "-mlp.pt"
        self.criterion = nn.MSELoss()               # simple loss - mean sqd. err

    def forward(self, x):
        '''
        Forward pass through the model. Get the value from
        the input sequence
        '''
        y_hat = self.seq(x)
        return y_hat

    def loss(self, y, y_hat):
        '''
        Determine loss between actual and calculated values
        '''
        return self.criterion(y, y_hat)

    def save(self):
        '''
        Write parameters to disk
        '''
        torch.save(self.state_dict(), self.model_file)

    @staticmethod
    def load(file):
        '''
        Create a new instance of an MLP with the saved parameters
        '''
        this = MLP(model_file=file)  # TODO - fix this error
        this.load_state_dict(torch.load(file))
        return this


class VAE(nn.ModuleDict):

    def __init__(self, num_inputs, ZDIMS, ZHID, model_file_prefix ):
        super(VAE, self).__init__()

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
        self.criterion = nn.MSELoss()
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
        # pass through VAE - encode then decode
        mu, log_var = self.encode(x)
        mu2 = self.reparameterize(mu, log_var)
        recon_x = self.decode(mu2)
        return recon_x, mu, log_var

    def loss(self, recon_x, x, mu, log_var):
        MSE = self.criterion(recon_x, x)
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return MSE + KLD

    def save(self):
        torch.save(self.state_dict(), self.model_file)

    @staticmethod
    def load(file):
        this = Model(model_file=file)
        this.load_state_dict(torch.load(file))
        return this
