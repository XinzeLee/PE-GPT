"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT
"""

import numpy as np




def locate(v, V):
    """
        Locate the moments when the device switching event occurs,
        Should be applied after sync
    """
    idx = [None]*4
    v0 = v[0]
    
    for i in range(1, len(v)+1):
        i = i % len(v)
        dV = v[i]-v0
        
        if (dV > V/2):
            if v0 < -V/2: idx[0] = i
            else: idx[1] = i
            if (dV > V*1.5):
                idx[1] = i
                
        elif (dV < -V/2):
            if v0 > V/2: idx[2] = i
            else: idx[3] = i
            if (dV < -V*1.5):
                idx[3] = i
                
        v0 = v[i]
    return idx


def eval_cs(pred, criterion="ipp"):
    """
        Evaluate the current stress based on the current waveforms
    """
    assert criterion in ["ipp", "irms"]
    
    if criterion == "ipp":
        current_stress = (pred.max(axis=1).ravel() - pred.min(axis=1).ravel())
    elif criterion == "irms":
        current_stress = (pred[..., 0] ** 2).mean(axis=1)
        
    return current_stress


def eval_zvzcs(inputs, pred, Vin, Vref, thres=1e-2):
    """
        Evaluate the soft switching metrics based on the current waveforms
    """
    ZVS = np.zeros((len(pred),))
    ZCS = np.zeros((len(pred),))
    
    for i in range(len(pred)):
        # locate the switching moments
        index_p = locate(inputs[i, :, 0], Vin)
        index_s = locate(inputs[i, :, 1], Vref)
        
        i_p = pred[i, index_p, 0]
        i_s = pred[i, index_s, 0]
        
        # ensure the current negatively flows through the device to ensure ZVS
        ZVS[i] = (i_p[:2] <= thres).sum() + (i_p[2:] >= -thres).sum() + \
                 (i_s[:2] >= -thres).sum() + (i_s[2:] <= thres).sum()
        ZCS[i] = (np.abs(i_p) <= thres).sum() + (np.abs(i_s) <= thres).sum()
    return ZVS, ZCS



