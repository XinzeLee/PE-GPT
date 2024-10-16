# -*- coding: utf-8 -*-
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

import matplotlib.pyplot as plt

from io import BytesIO
from ..optim.star_metric import locate




def plot_modulation(inputs, pred, Vin, 
                    Vref, PL, modulation):
    """
        plot the waveforms of the selected modulation with designed parameters
    """
    
    fig, ax1 = plt.subplots(figsize=(9, 6))
    ax2 = ax1.twinx() # dual axis
    ax1.plot(inputs[0, :, 0], label='Vp', color='r')
    ax1.plot(inputs[0, :, 1], label='Vs', color='g')
    index_p = locate(inputs[0, :, 0], Vin)
    index_s = locate(inputs[0, :, 1], Vref)
    ax1.scatter(index_p, inputs[0, index_p, 0])
    ax1.scatter(index_s, inputs[0, index_s, 1])
    
    ax2.plot(pred[0, :, 0], label='IL', color='b')
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.title(f"{modulation}:Uin={Vin:.0f} V, Uo={Vref:.0f} V, PL={PL:.0f} W")
    #plt.show()
    
    buf = BytesIO()
    fig.savefig(buf, format='png')
    # Important: close the figure to prevent memory leakage
    plt.close(fig)
    # move the buffer pointer to the beginning
    buf.seek(0)

    # return the buffer
    return buf



