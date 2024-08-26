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
from . import gui
from . import design_stages

# import functions
from .gui import build_gui, init_states, display_history
from .design_stages import design_flow, task_agent





__all__ = [
    # importable files
    gui,
    design_stages,
    
    
    # importable functions
    build_gui,
    init_states,
    display_history,
    
    design_flow,
    task_agent,
    
    
    # importable classes
    
    
    # importable variables
    # please directly import from the files #
    
    
    ]

