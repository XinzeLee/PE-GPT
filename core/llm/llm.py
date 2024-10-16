# -*- coding: utf-8 -*-
"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, and Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT

@reference:
    Following references are related to power electronics GPT (PE-GPT)
    1: PE-GPT: a New Paradigm for Power Electronics Design
        Authors: Fanfan Lin, Xinze Li (corresponding), Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma
        Paper DOI: 10.1109/TIE.2024.3454408
"""

import streamlit as st
import openai
from llama_index.core import VectorStoreIndex, ServiceContext, SimpleDirectoryReader
from llama_index.core.llms import ChatMessage, MessageRole
from llama_index.llms.openai import OpenAI
from llama_index.core.node_parser import SimpleNodeParser




def openai_init(openai_model=None, api_key=None, api_url=None):
    """
        initialize OpenAI, including api_key and api_base
    """    
    
    # please use your OpenAI's api key
    openai.api_key = "YOUR_API_KEY" if api_key is None else api_key
    api_base = "https://api.openai.com/v1/" if api_url is None else api_url
    openai.base_url = api_base
    
    # Provide the api key through the provided protal
    client = openai.OpenAI(api_key=openai.api_key, base_url=api_base)
    
    # model selection
    if "openai_model" not in st.session_state:
        st.session_state["openai_model"] = "gpt-4-0125-preview" if openai_model is None else openai_model
    
    return client


# Specialized multi-agents to handle different tasks with Retrieval Augmented Generation (RAG)
@st.cache_resource(show_spinner=False)
def rag_load(database_folder, llm_model, 
              temperature=None, chunk_size=None, system_prompt=None):
    """
        This function is the retrieval-augmented generation (RAG) for LLM
    """
    
    if chunk_size is None: chunk_size = 1024
    if temperature is None: temperature = 0.0
    
    with st.spinner(text="Loading and indexing  docs – hang tight! This should take 1-2 minutes."):
        
        docs = SimpleDirectoryReader(database_folder).load_data()
        node_parser = SimpleNodeParser.from_defaults(chunk_size=chunk_size)
        nodes = node_parser.get_nodes_from_documents(docs)
        service_context = ServiceContext.from_defaults(llm=OpenAI(model=llm_model, 
                                                                  temperature=temperature,
                                                                  system_prompt=system_prompt))
        index = VectorStoreIndex(nodes, service_context=service_context)
        return index


def get_msg_history():
    """
        get the message history 
    """
    messages_history = [ChatMessage(role=MessageRole.USER 
                                    if msg["role"] == "user" 
                                    else MessageRole.ASSISTANT, 
                                    content=msg["content"])
                        for msg in st.session_state.messages]
    return messages_history




