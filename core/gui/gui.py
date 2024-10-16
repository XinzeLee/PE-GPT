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


import streamlit as st
import numpy as np




def build_gui():
    """
        Create a graphical user interface (GUI) using streamlit
    """
    
    # Graphical User Interface
    st.set_page_config(page_title="PE-GPT", page_icon="💎", layout="centered",
                       initial_sidebar_state="auto", menu_items=None)
    st.title("Chat with the Power electronic robot🤖")
    st.info( "Hello, I am a robot specifically for power electronics design!", icon="🤟")
    
    with st.sidebar:
        st.markdown("<h1 style='color: #FF5733;'>PE-GPT (v2.0)</h1>", unsafe_allow_html=True)
        st.markdown('---')
        #st.markdown('\n- SPS:\n- EPS\n- DPS\n- TPS\n- 5DOF')
        st.markdown('\n- PE-GPT (v2.0) supports the design of modulation strategies for dual-active-bridge converters and circuit for buck converters.')
        st.markdown('\n- This repo highlights the software architecture with necessary submodules for your customized PE design tasks ')
        st.markdown('\n- @Reference: PE-GPT: a New Paradigm for Power Electronics Design ')
        st.markdown('\n- @Authors: Fanfan Lin, Xinze Li, et al. ')
        st.markdown('\n- @GitHub: https://github.com/XinzeLee/PE-GPT ')
        st.markdown('---')
        
    clear_button = st.sidebar.button('Clear Conversation',key='clear')
    # Create a scroll down selection box
    file_type = st.sidebar.selectbox("Select file type", ("vp", "vs", "iL"))
    # Create a file uploader
    uploaded_file = st.sidebar.file_uploader("Upload file", key="file_uploader")
    
    # A buttom to confirm file upload
    if st.sidebar.button("Confirm Upload"):
        if uploaded_file is not None:
            # load data for training
            upload_func(uploaded_file, file_type)
    
            # Notifications that file has been uploaded successfully
            st.sidebar.write(f"{file_type} file uploaded successfully.")
    
    # Provide initial guiding prompt after clicking the clear button
    with open('core/knowledge/prompts/prompt.txt', 'r') as file:
        content1 = file.read()
    with open('core/knowledge/prompts/init_reply.txt', 'r') as file:
        reply = file.read()
        
    if clear_button or ("messages" not in st.session_state):  # Initialize the chat messages history
        st.session_state.messages = [{"role": "user", "content": content1},
                                     {"role": "assistant", "content": reply},]


def upload_func(uploaded_file, file_type):
    """
        A function linked to the file upload button
    """
    
    if file_type == "vp":
        st.session_state.vp = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
    elif file_type == "vs":
        st.session_state.vs = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
    elif file_type == "iL":
        st.session_state.iL = np.loadtxt(uploaded_file, skiprows=1, delimiter=',')
    

def init_states(initial_values):
    """
        initialize st.session_state
    """
    for key, value in initial_values.items():
        if key not in st.session_state:
            st.session_state[key] = value


def display_history():
    """
        Display the historical chat messages
    """
    for msg in st.session_state.messages[2:]:  # Display the prior chat messages
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if "images" in msg:
                st.image(msg["images"])



