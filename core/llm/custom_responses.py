# -*- coding: utf-8 -*-
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




# Predefined responses for the modulation strategy and its diverse performances
def response(performances, modulation):
    ipp, nZVS, nZCS, P_required, pos = performances
    response = "No valid modulation strategy found."
    
    answer_format = """Under the {} modulation strategy, {}, 
    the number of switches that achieve zero-voltage turn-on is {:.0f}, 
    the number of switches that achieve zero-current turn-off is {:.0f}. 
    And the current stress performance is shown with the following figure. 
    At the power level (PL = {} W), the peak-to-peak current is {:.2f} A.""".replace("\n", "")
    
    if modulation == "SPS":
        D0 = pos[0]
        ps_str = f"the D0 is {D0:.3f}"
    elif modulation == "EPS1":
        D0, Din = pos[0], pos[1]
        ps_str = f"the D0 is {D0:.3f}, D2 is 1, the optimal D1 is designed to be {Din:.3f}"
    elif modulation == "EPS2":
        D0, Din = pos[0], pos[1]
        ps_str = f"the D0 is {D0:.3f}, D1 is 1, the optimal D2 is designed to be {Din:.3f}"
    elif modulation == "DPS":
        D0, Din = pos[0], pos[1]
        ps_str = f"the D0 is {D0:.3f}, the optimal D1 and D2 are designed to be {Din:.3f}"
    elif modulation == "TPS":
        D0, D1, D2 = pos[0], pos[1], pos[2]
        ps_str = f"the D0 is {D0:.3f}, the optimal D1 is {D1:.3f}, the optimal D2 is {D2:.3f}"
    elif modulation == "5DOF":
        D0, D1, D2, phi1, phi2 = pos[0], pos[1], pos[2], pos[3], pos[4]
        ps_str = f"the D0 is {D0:.3f}, the optimal D1 is {D1:.3f}, the optimal D2 is {D2:.3f}, the optimal phi1 is {phi1:.3f}, the optimal phi2 is {phi2:.3f}"
    response = answer_format.format(modulation, ps_str, nZVS, nZCS, P_required, ipp)
    
    return response




#######################################################
# Codes below are used for buck converters #
#######################################################



