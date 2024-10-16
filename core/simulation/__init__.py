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
from . import load_plecs

# import functions
from .load_plecs import open_plecs, dab_plecs

# import classes
from .load_plecs import PlecsThread




__all__ = [
    # importable files
    load_plecs,
    
    
    # importable functions
    open_plecs,
    dab_plecs,
    
    # importable classes
    PlecsThread,
    
    # importable variables
    # please directly import from the files #
    
    
    ]

