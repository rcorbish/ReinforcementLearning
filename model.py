# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.ModuleDict):

    def __init__(self, numInputs, numOutputs, model_file ):
        super(Model, self).__init__()

        self.fc1 = nn.Linear( numInputs, 96 ) 
        self.fc2 = nn.Linear( 96, 128 )
        self.fc3 = nn.Linear( 128, 96 )
        self.fc4 = nn.Linear( 96, 48 )
        self.fc5 = nn.Linear( 48, numOutputs )

        self.model_file = model_file

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

    def save( self ) :
        torch.save( self.state_dict(), self.model_file )

    @staticmethod
    def load( file="model.pt" ) :
        this = Model() 
        this.load_state_dict( torch.load(file) )
        return this
