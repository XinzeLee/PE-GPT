"""
Created on Thu Mar 21 07:50:04 2024

@author: XinzeLee
@github: https://github.com/XinzeLee/PANN

@reference:
    1: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Huai Wang, Xin Zhang, Hao Ma, Changyun Wen and Frede Blaabjerg
        Paper DOI: 10.1109/TIE.2024.3352119
    2: Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first author), Xin Zhang, Hao Ma and Frede Blaabjerg
        Paper DOI: 10.1109/TPEL.2024.3378184

"""

import torch
import copy
import numpy as np
import torch.nn as nn

from torch.utils.data import Dataset
from .pann_dab_vars import Tslen
from ..utils.pann_utils import evaluate, transform




class CustomDataset(Dataset):
    def __init__(self, states, inputs, targets):
        super(CustomDataset, self).__init__()
        self.states = states
        self.inputs = inputs
        self.targets = targets
        
    def __getitem__(self, index):
        return self.states[index], self.inputs[index], self.targets[index]
        
    def __len__(self):
        return len(self.states)
    

def train(model_pann, clamper, optimizer_pann, data_loader, 
          test_data, convert_to_mean=True, epoch=200):
    
    test_inputs, test_states = test_data
    
    loss_pann = nn.MSELoss()
    device = "cpu" # it is a waste to use gpu for this network
    
    
    loss_best_pann = np.inf
    best_pann_states = None
    
    model_pann = model_pann.to(device)
    for epoch in range(epoch):
        model_pann.train()
        
        #Forward pass
        total_loss = 0.
        for data in data_loader:
            """ 
                Logic is:
                input_ (full length) -> smooth_all -> PANN pred -> segment final Tslen*2 points -> sync
            """
            
            state, input_, target = data
            state, input_, target = state.to(device), input_.to(device), target.to(device)
            # state0 = state[:, :1] # should be zero to avoid learning the initial state
            state0 = torch.zeros(state.shape).to(device) # should be zero to avoid learning the initial state
            pred = model_pann.forward(input_, state0)
            Vin = 200
            pred, _ = transform(input_[:, -2*Tslen:], pred[:, -2*Tslen:], 
                                Vin, convert_to_mean=convert_to_mean)
            
            loss_train = loss_pann(pred, target)
            optimizer_pann.zero_grad()
            loss_train.backward()
            optimizer_pann.step()
            clamper(model_pann) # comment out this line if using pure data-driven model for dk
            total_loss += loss_train.item()
        # estimated_circuit = list(map(lambda x: round(x.item(), 7), model_pann.parameters()))
        # print("Estimations for circuit parameters: ", estimated_circuit)
        # print(f"Epoch {epoch}, Training loss {total_loss/len(data_loader)}")  
        
        if epoch % 1 == 0:
            *_, test_loss = evaluate(test_inputs[:, 1:], test_states, model_pann, 
                                     Vin, convert_to_mean=convert_to_mean)
            if test_loss < loss_best_pann:
                loss_best_pann, best_pann_states = test_loss, copy.deepcopy(model_pann.state_dict())
                # print(f"New loss is {best_pann}.")
                # print('-'*81)
                
    return best_pann_states
