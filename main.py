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

# from core.model_zoo.pann_dab_vars import *
from core.gui.gui import build_gui, init_states, display_history
from core.gui.design_stages import design_flow, task_agent
from core.llm.llm import openai_init, rag_load




# To run the PE-GPT, please run the following: 
# streamlit run main.py
if __name__ == "__main__":
    
    # flexible-response mode: use an LLM agent to enrich 
    # and enhance the PE expertise of predefined responses 
    FlexRes = True # whether to enable the flexible-response mode

    llm_model = "gpt-4-0125-preview"
    client = openai_init(openai_model=llm_model)
    build_gui()
        
    
    # Use Retrieval Augmented Generation (RAG) to embed customized knowledge base
    temperature, chunk_size, top_k = 0.1, 512, 7
    # AGENT 0 to provide insights and PE-specific reasoning for the selected modulations
    with open('core/knowledge/prompts/prompt.txt', 'r') as file:
        system_prompt = file.read()
    index0 = rag_load("core/knowledge/kb/database", llm_model, temperature=temperature, 
                       chunk_size=chunk_size, system_prompt=system_prompt)
    chat_engine0 = index0.as_chat_engine(chat_mode="context",similarity_top_k=top_k)
    # AGENT 1 specialized in modulation recommendation
    index1 = rag_load("core/knowledge/kb/database1", llm_model, temperature=temperature, 
                       chunk_size=chunk_size, system_prompt=system_prompt)
    chat_engine1 = index1.as_chat_engine(similarity_top_k=top_k)
    # AGENT 2 for self introduction
    index2 = rag_load("core/knowledge/kb/introduction", llm_model, temperature=temperature, 
                       chunk_size=chunk_size)
    chat_engine2 = index2.as_chat_engine(chat_mode="context",similarity_top_k=top_k)
    
    
    # Define an LLM agent to judge and keep track of the design stage/task
    agent_intent = task_agent()
    
    
    # define electrical variables that might be used 
    initial_values = {key:None for key in ['M', 'Uin', 'Uo', 'P', 
                                           'fs', 'vp', 'vs', 'iL']}
    # initialize st.session_state
    init_states(initial_values)
    
    
    # Display the historical chat messages
    display_history()


    # run the PE-GPT engine to conduct the design workflow
    agents = [chat_engine0, chat_engine1, chat_engine2, agent_intent]
    design_flow(agents, client, FlexRes=FlexRes)
    
    
    
    
    