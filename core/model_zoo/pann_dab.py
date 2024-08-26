"""
@reference: Temporal Modeling for Power Converters With Physics-in-Architecture Recurrent Neural Network, by Xinze Li, Fanfan Lin, et al.
@code-author: Xinze Li, Fanfan Lin
@github: https://github.com/XinzeLee/PE-GPT
"""

import onnxruntime
import streamlit as st

# initialize the ort inference session
model_pann_onnx = onnxruntime.InferenceSession("core/model_zoo/pann_dab.onnx", 
                                                providers=["CPUExecutionProvider"])
st.session_state["model_pann_onnx"] = model_pann_onnx