"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""
import torch
import numpy as np
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from collections import deque
from scipy import signal




def sync(vp, Vin):
    """
        Conduct synchronization
    """
    idx = 0
    for i in range(len(vp)-1):
        if (vp[i]<-Vin/2) and (vp[i+1]>=-Vin/2):
            idx = i+1
            break
    return idx


def create_vpvs(D0, D1, D2, Vin, Vref, 
                D1_cycle=0.5, D2_cycle=0.5):
    """
        Support these modulation strategies:
            1. Single phase shift (SPS)
            2. Double phase shift (DPS)
            3. Extended phase shift (EPS)
            4. Triple phase shift (TPS)
            5. Hybrid phase shift and duty cycle (5DOF)
    """
    
    D0 = D0+D1-D2
    # create a time array with the sampling period dt
    t = np.linspace(0, Ts, round(Ts/dt), endpoint=False)
    
    # create switching functions
    s_pri = signal.square(2*np.pi/Ts*t, D1_cycle)
    s_pri2 = deque(-s_pri)
    s_pri2.rotate(int(np.ceil(np.round(D1*Ts/2/dt, 5))))
    
    s_sec = deque(signal.square(2*np.pi/Ts*t, D2_cycle))
    s_sec.rotate(int(np.ceil(np.round(D0*Ts/2/dt, 5))))
    s_sec2 = deque(-np.array(s_sec))
    s_sec2.rotate(int(np.ceil(np.round(D2*Ts/2/dt, 5))))
    
    # create high-frequency ac waveforms
    vp = (np.array(s_pri)+np.array(s_pri2)).clip(-1, 1)*Vin
    vs = (np.array(s_sec)+np.array(s_sec2)).clip(-1, 1)*Vref
    
    # extend the length of waveforms
    vp = np.tile(vp, (round(Tsim/Ts),))
    vs = np.tile(vs, (round(Tsim/Ts),))
    
    # perform synchronization
    idx = sync(vp, Vin)
    vp = vp[idx:Tslen*(round(Tsim/Ts)-1)+idx]
    vs = vs[idx:Tslen*(round(Tsim/Ts)-1)+idx]
    return vp, vs


def transform(input_, pred, Vin, convert_to_mean=False):
    """
        Perform synchronization, which extends to general modulation modeling
    """
    
    if isinstance(input_, torch.Tensor):
        pred_o = torch.zeros(input_.shape[0], Tslen, 
                             pred.shape[-1]).to(input_.device)
        input_o = torch.zeros(input_.shape[0], Tslen, 
                              input_.shape[-1]).to(input_.device)
    elif isinstance(input_, np.ndarray):
        pred_o = np.zeros((input_.shape[0], Tslen, 
                           pred.shape[-1]), np.float32)
        input_o = np.zeros((input_.shape[0], Tslen, 
                            input_.shape[-1]), np.float32)
    
    for i in range(input_.shape[0]):
        vp = input_[i, :, 0]
        idx = sync(vp, Vin)
        pred_o[i] = pred[i, idx:idx+Tslen]
        input_o[i] = input_[i, idx:idx+Tslen]
        if convert_to_mean:
            pred_o[i, :, 0] -= pred_o[i, :, 0].mean() # eleminate DC bias for steady-state waveforms
    return pred_o, input_o


def get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref):
    """
        Expert system: store the knowledge to generate key switching waveforms vp and vs
    """
    
    inputs = []
    
    for _D0, _D1, _D2, _phi1, _phi2 in zip(D0, D1, D2, phi1, phi2):
        vp_, vs_ = create_vpvs(_D0, _D1, _D2, Vin, Vref, 
                              D1_cycle=0.5+_phi1/2, D2_cycle=0.5+_phi2/2)
        _input = np.concatenate([vp_[None, :, None], 
                                  vs_[None, :, None]], axis=-1)
        inputs.append(_input)
    return np.concatenate(inputs, axis=0)


def evaluate(inputs, targets, model_pann, 
             Vin, convert_to_mean=True):
    """
        Evaluate all for PANN using PyTorch
    """
    
    model_pann = model_pann.to("cpu")
    model_pann.eval()
    
    with torch.no_grad():
        
        if targets is None:
            state0 = torch.zeros((inputs.shape[0], 1, 1))
        else:
            state0 = targets[:, 0:1]
        pred = model_pann.forward(inputs, state0)
        
        pred, inputs = transform(inputs[:, -2*Tslen:], pred[:, -2*Tslen:], Vin,
                                 convert_to_mean=convert_to_mean)
        if targets is None:
            return pred, inputs
        test_loss = (targets[:, 1:]-pred).abs().mean().item()
        return pred, inputs, test_loss


def evaluate_onnx(inputs, targets, model_pann_onnx, 
                  Vin, convert_to_mean=True):
    """
        Evaluate all for PANN using ONNX inference engine
    """
    
    key1, key2 = [key.name for key in model_pann_onnx.get_inputs()]
    ort_inputs = {key: None for key in [key1, key2]}
    
    
    if targets is None:
        state0 = np.zeros((inputs.shape[0], 1, 1)).astype(np.float32)
    else:
        state0 = targets[:, 0:1]

    pred = []
    for j in range(inputs.shape[1]//seqlen_onnx):
        ort_inputs[key1] = inputs[:, j*seqlen_onnx:(j+1)*seqlen_onnx]
        ort_inputs[key2] = state0
        ort_outs = model_pann_onnx.run(None, ort_inputs)[0]
        state0 = ort_outs[:, -1:]
        pred.append(ort_outs)
    pred = np.concatenate(pred, axis=1)
        
    pred, inputs = transform(inputs[:, -2*Tslen:], pred[:, -2*Tslen:], Vin,
                             convert_to_mean=convert_to_mean)
    if targets is None:
        return pred, inputs
    test_loss = np.abs(targets[:, 1:]-pred).mean()
    return pred, inputs, test_loss

from ..model_zoo.pann_dab_vars import Ts, dt, Tsim, Tslen, seqlen_onnx



