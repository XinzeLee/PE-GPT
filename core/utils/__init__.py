# Manage the importable files, functions, classes, and variables
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

# import files
from . import pann_utils
from . import plots

# import functions
from .plots import plot_modulation
from .pann_utils import sync, create_vpvs, transform, \
    get_inputs, evaluate_onnx




__all__ = [
    # importable files
    pann_utils,
    plots,
    
    
    # importable functions
    plot_modulation,
    
    sync,
    create_vpvs,
    transform,
    get_inputs,
    evaluate_onnx,
    
    
    # importable classes
    
    
    # importable variables
    # please directly import from the files #
    
    
    ]

