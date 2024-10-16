"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import numpy as np
import pyswarms as ps
import streamlit as st

from ..optim import obj_func
from ..utils.plots import plot_modulation




def optimize_cs(nums, model_pann, PL, Vin, Vref, 
                modulation, upper_bound, lower_bound, 
                bh_strategy, vh_strategy, with_ZVS=False):
    """
        Optimize the current stress through particle swarm optimization algorithm
    """
    upper_bounds = np.array(upper_bound)
    lower_bounds = np.array(lower_bound)
    PSO_optimizer = ps.single.GlobalBestPSO(n_particles=50, dimensions=len(upper_bounds), 
                                            bounds=(lower_bounds, upper_bounds),
                                            options={'c1': 2.05, 'c2': 2.05, 'w':0.9},
                                            bh_strategy=bh_strategy,
                                            #velocity_clamp=(lower_bounds*0.4, upper_bounds*0.4),
                                            velocity_clamp=None,
                                            vh_strategy=vh_strategy,
                                            oh_strategy={"w": "lin_variation"})
    cost, pos = PSO_optimizer.optimize(obj_func.obj_func, nums,
                                        model_pann=model_pann,
                                        PL=PL, Vin=Vin, Vref=Vref,
                                        modulation=modulation,
                                        with_ZVS=with_ZVS) # pytorch objective function
#     cost, pos = PSO_optimizer.optimize(obj_func.obj_func_onnx, nums,
#                                         model_pann_onnx=model_pann,
#                                         PL=PL, Vin=Vin, Vref=Vref,
#                                         modulation=modulation,
#                                         with_ZVS=with_ZVS) # ONNX objective function
    return cost, pos


def optimize_mod_dab(Vin, Vref, PL, modulation):
    """
        Optimize the modulation parameters for DAB converters with specified 
        operating conditions and recommended modulation strategy
    """
    
    # define hyperparameters for optimizers
    bh_strategy = "periodic"
    vh_strategy = "unmodified"
    num_iter = 50
    st.write(f"Recommended Modulation is: {modulation}")
    
    # define the searching boundaries
    if modulation == "SPS":
        upper_bound = [0.35]
        lower_bound = [-0.2]
        
    elif modulation in ["EPS", "DPS"]:
        upper_bound = [0.35, 1.0]
        lower_bound = [-0.2, 0.6]
        if modulation == "EPS":
            if Vin > n*Vref: modulation = "EPS1"
            else: modulation = "EPS2"
        
    elif modulation == "TPS":
        num_iter = 100
        upper_bound = [0.35, 1.0, 1.0]
        lower_bound = [-0.2, 0.6, 0.6]
        
    elif modulation == "5DOF":
        num_iter = 200
        upper_bound = [0.35, 1.0, 1.0, 0.2, 0.2]
        lower_bound = [-0.2, 0.6, 0.6, -0.2, -0.2]
        
    else:
        st.write(f"There is something wrong with the recommended strategy: {modulation}.")
        
    # conduct the optimization algorithm
    obj, optimal_x = optimize_cs(num_iter, model_pann, PL, Vin, Vref,
                                 modulation, upper_bound, lower_bound,
                                 bh_strategy, vh_strategy, with_ZVS=True)
    
    # evaluate all performance metrics (pytorch)
    ipp, P_pred, pred, inputs, \
    ZVS, ZCS, penalty = obj_func.obj_func(optimal_x[None], model_pann, PL, Vin, Vref, 
                                          with_ZVS=True, modulation=modulation, return_all=True)
    
    # evaluate all performance metrics (ONNX)
#     ipp, P_pred, pred, inputs, \
#     ZVS, ZCS, penalty = obj_func.obj_func_onnx(np.tile(optimal_x[None], (50, 1)), model_pann_onnx, PL, Vin, 
#                                                 Vref, with_ZVS=True, modulation=modulation, return_all=True)
    
    pos = list(map(lambda x: round(x, 3), optimal_x))
    plot = plot_modulation(inputs, pred, Vin, Vref, PL, modulation)
    
    return ipp[0], ZVS[0], ZCS[0], PL, pos, plot, modulation

from ..model_zoo.pann_dab import model_pann
# from ..model_zoo.pann_dab import model_pann_onnx
from ..model_zoo.pann_dab_vars import n
