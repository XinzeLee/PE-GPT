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

@reference:
    Following references are related to physics-in-architecture recurrent neural networks (PA-RNN)
    1: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network
        Authors: Xinze Li, Fanfan Lin (corresponding), Huai Wang, Xin Zhang, Hao Ma, Changyun Wen and Frede Blaabjerg
        Paper DOI: 10.1109/TIE.2024.3352119
    2: Data-Light Physics-Informed Modeling for the Modulation Optimization of a Dual-Active-Bridge Converter
        Authors: Xinze Li, Fanfan Lin (corresponding and co-first), Xin Zhang, Hao Ma and Frede Blaabjerg
        Paper DOI: 10.1109/TPEL.2024.3378184
    3: STAR: One-Stop Optimization for Dual-Active-Bridge Converter With Robustness to Operational Diversity
        Authors: Fanfan Lin, Xinze Li (corresponding), Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/JESTPE.2024.3392684
    4: A Generic Modeling Approach for Dual-Active-Bridge Converter Family via Topology Transferrable Networks
        Authors: Xinze Li, Fanfan Lin (corresponding), Changjiang Sun, Xin Zhang, Hao Ma, Changyun Wen, Frede Blaabjerg, and Homer Alan Mantooth
        Paper DOI: 10.1109/TIE.2024.3406858
"""

# import files
from . import pann_dab
from . import pann_dab_vars
from . import pann_net
from . import pann_train

# import variables
# from pann_dab_vars import *




__all__ = [
    # importable files
    pann_dab,
    pann_dab_vars,
    pann_net,
    pann_train,
    
    # importable functions

    
    # importable classes
    
    
    # importable variables
    # please directly import from the files #
    
    
    ]

