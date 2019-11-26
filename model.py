# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.ModuleDict):

    def __init__(self, numInputs, numOutputs ):
        super(Model, self).__init__()

        self.fc1 = nn.Linear( numInputs, 128 ) 
        self.fc2 = nn.Linear( 128, 128 )
        self.fc3 = nn.Linear( 128, 96 )
        self.fc4 = nn.Linear( 96, 48 )
        self.fc5 = nn.Linear( 48, numOutputs )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
