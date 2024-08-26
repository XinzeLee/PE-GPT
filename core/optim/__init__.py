# Manage the importable files, functions, classes, and variables
"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT


@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 
"""

# import files
from . import obj_func
from . import optimizers
from . import star_metric

# import functions
from .optimizers import optimize_cs, optimize_mod_dab
from .star_metric import locate, eval_cs, eval_zvzcs





__all__ = [
    # importable files
    obj_func,
    optimizers,
    star_metric,
    
    
    # importable functions
    optimize_cs,
    optimize_mod_dab,
    
    locate,
    eval_cs,
    eval_zvzcs,
    
    
    # importable variables
    # please directly import from the files #
    
    
    ]

