# Manage the importable files, functions, classes, and variables
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

# import files
from . import llm
from . import custom_responses

# import functions
from .llm import openai_init, rag_load, get_msg_history
from .custom_responses import response





__all__ = [
    # importable files
    llm,
    custom_responses,
    
    
    # importable functions
    openai_init,
    rag_load,
    get_msg_history,
    
    response,
    
    
    # importable classes
    
    
    # importable variables
    # please directly import from the files #
    
    
    ]

