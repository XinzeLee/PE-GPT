"""
@reference: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network, by Xinze Li, Fanfan Lin, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT
"""

# Define global variables for PA-RNN of DAB
# Define variables for PA-RNN of DAB
Tslen = 250
fs = 5e4
Ts = 1/fs
dt = Ts/Tslen
Tsim = Ts*10
seqlen_onnx = 1 # sequence length of the pann model deployed in ONNX


# (Initial) Circuit parameters of DAB
n = 1
RL = 500e-3
Lr = 60e-6



