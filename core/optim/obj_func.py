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

import copy
import torch
import numpy as np

from ..utils.pann_utils import get_inputs, evaluate, evaluate_onnx
from ..optim.star_metric import eval_cs, eval_zvzcs




def obj_func(x, model_pann, PL, Vin, Vref,
                  modulation="5DOF", with_ZVS=False,
                  return_all=False, thres_ZVS=1e-2):
    """
        Objective function to optimize current stress and 
        soft switching range for the given power level
        
        /* This is a demo objective function, customization */
        /* is needed for other objective combinations.      */
    """
    
    if modulation == "5DOF":
        D0, D1, D2, phi1, phi2 = x.T.tolist()
    elif modulation == "TPS":
        D0, D1, D2 = x.T.tolist()
        phi1, phi2 = [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "DPS":
        D0, D1 = x.T.tolist()
        D2, phi1, phi2 = D1, [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "EPS1":
        D0, D1 = x.T.tolist()
        D2, phi1, phi2 = [1.0] * len(D0), [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "EPS2":
        D0, D2 = x.T.tolist()
        D1, phi1, phi2 = [1.0] * len(D0), [0.0] * len(D0), [0.0] * len(D0)
    elif modulation == "SPS":
        D0 = x.flatten().tolist()
        D1, D2, phi1, phi2 = [1.0] * len(D0), [1.0] * len(D0), [0.0] * len(D0), [0.0] * len(D0)


    model_pann.eval()
    with torch.no_grad():
        inputs = get_inputs(D0, D1, D2, phi1, phi2, Vin, Vref).astype(np.float32)
        inputs = torch.FloatTensor(inputs)
        pred, inputs = evaluate(inputs, None, model_pann, Vin, convert_to_mean=True)
        P_pred = (inputs[..., 0] * pred[..., 0]).mean(dim=1).numpy()
        pred, inputs = pred.numpy(), inputs.numpy()
#         pred, inputs = evaluate_onnx(inputs, None, model_pann_onnx, 
#                                      Vin, convert_to_mean=True) # use onnx model for inference
#         P_pred = (inputs[..., 0] * pred[..., 0]).mean(axis=1)
        ipp = eval_cs(pred, criterion="ipp")

    if with_ZVS:
        ZVS, ZCS = eval_zvzcs(inputs, pred, Vin, Vref, thres_ZVS)
    else:
        ZVS, ZCS = 0, 0  # do not consider ZVS and ZCS performances

    penalty = np.zeros((len(pred),)) # penalize the breach of required power levels
    P_threshold = 5.
    idx = np.abs(P_pred - PL) > P_threshold
    
    # Penalization method 1 
    # penalty[idx] = 100.0
    # Penalization method 2
    penalty[idx] = (np.abs(P_pred[idx] - PL) - P_threshold) * 2
    ipp_origin = copy.deepcopy(ipp)
    ipp[~idx] = ipp[~idx] * PL / P_pred[~idx]
    
    if return_all:
        return ipp_origin, P_pred, pred, inputs, ZVS, ZCS, penalty
    return ipp-(ZVS+ZCS)*5+penalty



