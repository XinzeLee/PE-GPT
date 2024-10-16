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
from torch import nn




class WeightClamp(object):
    """
    A simple calculator class to perform basic arithmetic operations.

    Attributes:
    -----------
    last_result : float
        Stores the result of the last operation.

    Methods:
    --------
    add(a, b):
        Returns the sum of two numbers.
    subtract(a, b):
        Returns the difference between two numbers.
    """
    
    
    """
        Clamp the weights to specified limits
        arguments: 
            arg::attrs -> a list of attributes in 'str' format for the respective modules
            arg::limits -> a list of limits for the respective modules, 
                            where limits[idx] follows [lower bound, upper bound]
    """
    def __init__(self, attrs, limits):
        """
        Initializes the Calculator with a default last_result of 0.0.
        """
        
        self.attrs = attrs
        self.limits = limits
    
    def __call__(self, module):
        for i, (attr, limit) in enumerate(zip(self.attrs, self.limits)):
            w = getattr_(module, attr).data
            w = w.clamp(limit[0], limit[1])
            getattr_(module, attr).data = w
            
            
def getattr_(module, attr):
    # recurrence to the final layer of attributes
    attrs = attr.split('.')
    if len(attrs) == 1: return getattr(module, attrs[0])
    else: return getattr_(getattr(module, attrs[0]), ".".join(attrs[1:]))
        
    
class PANN(nn.Module):
    """
        Define the generic physics-in-architecture recurrent neural network (PA-RNN) structure
        
        References:
            1: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
                Paper DOI: 10.1109/TIE.2024.3352119
            2: Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter
                Paper DOI: 10.1109/TPEL.2024.3378184
    """
    
    def __init__(self, cell, **kwargs):
        super(PANN, self).__init__(**kwargs)
        self.cell = cell

    def forward(self, inputs, x):
        outputs = []
        _x = x[:, 0]
        for t in range(inputs.shape[1]):
            state_next = self.cell.forward(inputs[:, t, :], _x)
            _x = state_next
            outputs.append(_x)
        return torch.stack(outputs, dim=1)
    
    
class EulerCell_DAB(nn.Module):
    """
        Define Implicit Euler Recurrent Cell of PANN for Conventional DAB with Single Inductance L
    """
    
    def __init__(self, dt, Lr, RL, n, **kwargs):
        super(EulerCell_DAB, self).__init__(**kwargs)
        self.dt = dt
        self.Lr = nn.Parameter(torch.Tensor([Lr]))
        self.RL = nn.Parameter(torch.Tensor([RL]))
        self.n = nn.Parameter(torch.Tensor([n]))
        
    def forward(self, inputs, states):
        iL_next = (self.Lr/(self.Lr+self.RL*self.dt))*states[:, 0]+\
                    (self.dt/(self.Lr+self.RL*self.dt))*(inputs[:, 0]-self.n*inputs[:, 1])
        return iL_next[:, None]
    
    
class EulerCell_Buck(nn.Module):
    """
        Define Explicit Euler Recurrent Cell of PANN for Conventional Buck
    """
    
    def __init__(self, Vin, dt, L, Co, Ro, **kwargs):
        super(EulerCell_Buck, self).__init__(**kwargs)
        self.Vin = nn.Parameter(torch.Tensor([Vin]))
        self.dt = dt
        self.L = nn.Parameter(torch.Tensor([L]))
        self.Co = nn.Parameter(torch.Tensor([Co]))
        self.Ro = nn.Parameter(torch.Tensor([Ro]))

    def forward(self, inputs, states):
        # explict Euler method
        # inputs : represent s_pri, respectively
        # states : represent iL, vo, respectively
        vo = states[:, 1]
        va = self.Vin * inputs[:, 0] # terminal voltage
        idx = (inputs[:, 0] == 0) & (states[:, 0] <= 0) # evaluate discrete conduction mode
        va[idx] = vo[idx]
        iL_next = states[:, 0] + self.dt / self.L * (va - vo) # physics of iL
        iL_next = torch.relu(iL_next) # torch.relu to consider DCM
        vC_next = states[:, 1] + self.dt / self.Co * (states[:, 0] - vo / self.Ro) # physics of vC
        return torch.stack((iL_next, vC_next), dim=1)
