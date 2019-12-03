# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F

ZDIMS = 5
ZHID = 10
class Model(nn.ModuleDict):

    def __init__(self, numInputs, numOutputs, hidden_sizes, model_file ):
        super(Model, self).__init__()

        # VAE section
        self.encoder = nn.Sequential( 
                nn.Linear(numInputs, ZHID ),
                nn.ReLU( True )
        )
        self.encoderMu = nn.Linear(ZHID, ZDIMS)  # mu layer
        self.encoderLogVar = nn.Linear(ZHID, ZDIMS)  # logvariance layer
        # VAE bottleneck

        self.decoder = nn.Sequential( 
            nn.Linear(ZDIMS, ZHID),
            nn.ReLU( True ),
            nn.Linear(ZHID, numInputs ),
            nn.Sigmoid()
        )

        # MLP section
        layers = []
        ix = 0 
        prev_layer = numInputs 
        for h in hidden_sizes :
            layers.append( nn.Linear( prev_layer, h ) ) 
            layers.append( nn.ReLU( True ) ) 
            prev_layer = h

        layers.append( nn.Linear( h, numOutputs ) ) 
        self.seq = nn.Sequential( *layers )
        self.model_file = model_file
        self.criterion = nn.MSELoss()


    def encode( self, x ) :
        h1 = self.encoder(x)
        mu = self.encoderMu(h1)
        logvar = self.encoderLogVar(h1)
        return mu, logvar
    
    def reparameterize( self, mu, logvar ) :
        if self.training:
            std = torch.exp( logvar * 0.5 )  # type: torch.tensor
            eps = torch.randn_like( std )
            return eps * std + mu 
        else:
            return mu

    def decode(self, z ):
        return self.decoder( z )

    def forward(self, x):
        # first pass through VAE
        mu, logvar = self.encode( x )
        mu2 = self.reparameterize(mu, logvar)
        z = self.decode(mu2)
        # Then MLP to get categorization
        yhat = self.seq( z )
        return yhat, z, mu, logvar


    def loss( self, yhat, y, z, x, mu, logvar ) :
        BCE = F.binary_cross_entropy(z, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        # BCE tries to make our reconstruction as accurate as possible
        # KLD tries to push the distributions as close as possible to unit Gaussian
        return BCE + KLD + self.criterion( y, yhat )


    def save( self ) :
        torch.save( self.state_dict(), self.model_file )

    @staticmethod
    def load( file="model.pt" ) :
        this = Model( model_file=file ) 
        this.load_state_dict( torch.load(file) )
        return this

