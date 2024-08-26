"""
@functionality
    Mainly used for your custom design workflow.
    The design stages are defined in each individual function, 
    and another LLM agent is used to classify stage.
    Main hub to build GUI, LLM, interact with model zoo, simulation validation.


@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT
"""

import re
import streamlit as st

from ..simulation import load_plecs
from ..llm import custom_responses
from ..optim import optimizers
from ..llm.llm import get_msg_history

from llama_index.core.tools import FunctionTool
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI




# Task Indicators
# Task-0: Initialize the design process and provide guidance to users
def init_design_():
    """
        Main purpose: Initialize the design process and provide guidance on the design steps to users
        Usage: Only used when the user requests for the design.
    """
    return "Task 0" # just an indicator

def init_design(chat_engine, prompt, messages_history):
    """
        Executables of Task 0
    """
    with st.spinner("Thinking..."):
        response = chat_engine.chat(prompt, messages_history) # AGENT 1 for modulation recommendation
        st.write(response.response) # write the response into the GUI display
        return response.response


# Task Indicators
# Task-1: Understand user's requirements and Recommend modulation
def recommend_modulation_():
    """
        Main purpose: Understand user's requirements and recommend suitable modulation strategies.
                      The performances can be efficiency, power loss, current stress, soft switching, easy implementation, etc.
                      Pay attention: If the user mentions any modulation performances or objectives, this function should be called!!!
        Usage: User can modify their requirements anytime during the interactions with PE-GPT 
    """
    return "Task 1" # just an indicator

def recommend_modulation(chat_engine, prompt, messages_history):
    """
        Executables of Task 1
    """
    
    with st.spinner("Thinking..."):
        response = chat_engine.chat(prompt, messages_history) # AGENT 1 for modulation recommendation
        st.write(response.response)
        
        modulation_methods = ["SPS", "DPS", "EPS", "TPS", "5DOF"]
        prompt = f"""Based on your response quoted in '' below, which strategy in 
        the list {modulation_methods} do you recommend? 
        Attention: Only output the recommended strategy from the list, nothing else!!!
        """.replace("\n", "")+f"'{response.response}'"
        response2 = chat_engine.chat(prompt) # AGENT 1 for modulation recommendation
        
        recommended_mod = "TPS"
        # capture the recommended modulation and its location
        for method in modulation_methods:
            index = response2.response.lower().find(method.lower())
            if index != -1:
                recommended_mod = method
                break
        # set st.session_state.M if a recommendation has been given
        st.session_state.M = recommended_mod
        
        messages = [{"role": "assistant", "content": response.response},]
    return messages


# Task Indicators
# Task-2: Evaluate the waveforms and various converter performances by interacting with Model Zoo 
# given the operating conditions specified by users
def evaluate_dab_():
    """
        Main purpose: Evaluate the waveforms and various converter performances given the operating conditions specified by users
        Usage: Apply after user has provided the converter operating conditions, including input voltage Uin, output voltage Uo, power level PL
        Pay attention: If the user specifies the operating conditions (like input and output voltages, and power values), this function should be called!!!
    """
    return "Task 2" # just an indicator

def evaluate_dab(chat_engine, prompt, messages_history):
    """
        Executables of Task 2
    """
    re_specs = re.compile(r".*\[\D*(\d+)\D*\,\D*(\d+)\D*\,\D*(\d+)\D*\]")
    with st.spinner("Thinking..."):
        prompt = prompt+"\n Please be really careful about the response format for this request!!!! In the form of [Uin, Uo, PL]!!!"
        response = chat_engine.chat(prompt, messages_history)
        
        matched = re_specs.findall(response.response)
        if len(matched):
            st.session_state.Uin, st.session_state.Uo, \
                st.session_state.P = map(float, matched[0])
        
        Uins, Uos = [st.session_state.Uin]*2, [st.session_state.Uo]*2
        Ps = [st.session_state.P, st.session_state.P]
        Ms = [st.session_state.M, "SPS"]
        messages = []
        
        for Uin, Uo, P, M in zip(Uins, Uos, Ps, Ms):
            *performances, plot, updated_M = optimizers.optimize_mod_dab(Uin, Uo, P, M)
            if M == st.session_state.M:
                st.session_state["pos"] = performances[-1][1:] # get the optimized modulation parameters
            response = custom_responses.response(performances, updated_M)
            
            st.write(response)
            st.image(plot)
            messages.append({"role": "assistant", "content": response,"images": [plot]})
    return messages


# Task Indicators
# Task-3: Verify the designed modulation in commercial simulation tools
def simulation_verification_():
    """
        Main purpose: Open the integrated simulation models and conduct simulation to validate the designed modulation.
        Usage: User can choose to conduct simulation after the modulation strategy has been designed
    """
    return "Task 3" # just an indicator

def simulation_verification():
    """
        Executables of Task 3
    """
    with st.spinner("Waiting... PLECS is starting up..."):
        load_plecs.dab_plecs(st.session_state.M, st.session_state.Uin, st.session_state.Uo, 
                             st.session_state.P, *st.session_state.pos)
        reply = "The PLECS simulation is running... Complete! You can now verify if the design is reasonable by observing the simulation waveforms."
        st.write(reply)
        messages = [{"role": "assistant", "content": reply}]
    return messages


# Task Indicators
# Task-4: Introduction to PE-GPT
def pe_gpt_introduction_():
    """
        Main purpose: Understand user's requirements and recommend suitable modulation strategies.
        Usage: User can modify their requirements anytime during the interactions with PE-GPT 
    """
    return "Task 4" # just an indicator

def pe_gpt_introduction(chat_engine, prompt):
    """
        Executables of Task 4
    """
    with st.spinner("Thinking..."):
        response = chat_engine.chat(prompt) # AGENT 2 for brief introduction to PE-GPT
        st.write(response.response)
        messages = [{"role": "assistant", "content": response.response},]
    return messages


# Task Indicators
# Task-5: Fine-tune/train the PA-RNN model
def train_pann_():
    """
        Main purpose: Train or fine-tune the PA-RNN models in model zoo
        Usage: After all the datasets are provided, the PA-RNN model will be trained or fine-tuned
    """
    return "Task 5" # just an indicator

def train_pann():
    """
        Executables of Task 5
    """
    st.write("The training starts... Nothing is defined.")
    # The training of PANN models will be progressively open-sourced
    messages = [{"role": "assistant", "content": ""},]
    return messages


# Task Indicators
# Other tasks to be defined
def other_tasks_(*args, **kwargs):
    """
        Main purpose: Perform all other tasks through a general LLM
        Usage: If tasks not defined in all the above tasks, use a general LLM
    """
    return None # just an indicator

def other_tasks(client):
    """
        Executables of Other Tasks
    """
    with st.spinner("Loading..."):
        st.write("Entering the last block")
        response = client.chat.completions.create(
            model=st.session_state["openai_model"],
            messages=[
                {"role": "system", "content": """You are now an expert in the power electronics industry, 
                                             and you are proficient in optimal design of buck converter. Please answer the questions 
                                             in a warm, positive and friendly manner. Keep your answer less than 150 words! Make sure 
                                             your answers are professional and accurate -- don't hallucinate."""},
                *[{"role": msg["role"], "content": msg["content"]} 
                  for msg in st.session_state.messages] # provide all historical chat messages
                    ], stream=False,)
        st.write(response.choices[0].message.content) # get the first choice
        messages = [{"role": "assistant", "content": response.choices[0].message.content}]
        return messages


# Task Indicators
# Your customized tasks to be defined
def custom_tasks_():
    """
        Your customized tasks can be defined here.
        Please follow the template above.
        Right now, the following tasks are NOT supportedd:
            train PA-RNN, change circuit parameter in PA-RNN, circuit design for buck, etc.
    """
    pass

def custom_tasks():
    pass




def design_flow(agents, general_client, FlexRes=True):
    """
        This is your customized design workflow
    """
    
    chat_engine0, chat_engine1, chat_engine2, agent_intent = agents
    
    # User textual input block for queries
    if prompt := st.chat_input("Your request:"):  # Prompt of user inputs and save to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # save the historical messages into a list, to ensure that PE-GPT knows the chat history (LLM is memoryless in nature)
        messages_history = get_msg_history()
        # messages_history = [] # if no historical messages are used
        
        
        # The LLM agents are responsible for the following defined design tasks
        with st.chat_message("assistant"):
            
            response = agent_intent.chat(f"""Please call the corresponding function based the user's Request given in square brackes '[]' 
                                         and Listen Carefully!! Only ONE function closest to the function descriptions should be called!!!!
                                         User's Request: [{prompt}]""".replace('\n', ''))
                                         # +"\n\nThe detailed function descriptions are defined below."
                                         # +"\n".join([description_task0, description_task1, description_task2,
                                         #             description_task3, description_task4, description_other_tasks])
            
            if len(response.sources) >= 1: # if any function has been triggered
                if None in [item.raw_output for item in response.sources]: messages = other_tasks(general_client)
                else:
                    task = response.sources[0].raw_output # conduct the first matched task
                    
                    args = ()
                    if task == "Task 0":
                        kwargs = {"chat_engine": chat_engine1, "prompt": prompt, 
                                  "messages_history": messages_history}
                        response_pe = init_design(*args, **kwargs)
                        messages = [{"role": "assistant", "content": response_pe},]
                        
                    elif task == "Task 1":
                        kwargs = {"chat_engine": chat_engine1, "prompt": prompt, 
                                  "messages_history": messages_history}
                        messages = recommend_modulation(*args, **kwargs)
                        
                    elif task == "Task 2":
                        kwargs = {"chat_engine": chat_engine0, "prompt": prompt, 
                                  "messages_history": messages_history}
                        messages = evaluate_dab(*args, **kwargs) # these messages are predefined
                        if FlexRes:
                            prompt = """Attention!!! Now I have evaluated the current stress and soft switching
                            performances of the recommended modulation and the conventional SPS strategy, as the contents 
                            shown below. Please refer to your expertise for DAB modulations, completely rewrite the 
                            contents and provide more power electronics insights.""".replace("\n", "") + "\n" +\
                                "'"+"\n".join(item["content"] for item in messages)+"'"
                            response = chat_engine0.chat(prompt, messages_history)
                            st.write(response.response)
                            messages = [{"role": "assistant", "content": response.response},]
                        
                    elif task == "Task 3":
                        kwargs = {}
                        messages = simulation_verification(*args, **kwargs)
                        
                    elif task == "Task 4":
                        kwargs = {"chat_engine": chat_engine2, "prompt": prompt}
                        messages = pe_gpt_introduction(*args, **kwargs)
                        
                    elif task == "Task 5":
                        kwargs = {}
                        messages = train_pann(*args, **kwargs)
                    
            else: # no function has been triggered
                messages = other_tasks(general_client)
                    
        for msg in messages:
            st.session_state.messages.append(msg) # Append the response to the message history


def task_agent():
    """
        Define an LLM agent to judge and keep track of the design stage/task
    """
    
    init_design_tool = FunctionTool.from_defaults(fn=init_design_)
    recommend_modulation_tool = FunctionTool.from_defaults(fn=recommend_modulation_)
    evalualte_dab_tool = FunctionTool.from_defaults(fn=evaluate_dab_)
    simulation_verification_tool = FunctionTool.from_defaults(fn=simulation_verification_)
    pe_gpt_introduction_tool = FunctionTool.from_defaults(fn=pe_gpt_introduction_)
    train_pann_tool = FunctionTool.from_defaults(fn=train_pann_)
    other_tasks_tool = FunctionTool.from_defaults(fn=other_tasks_)

    llm = OpenAI(model="gpt-3.5-turbo-1106")
    agent = OpenAIAgent.from_tools(
        [init_design_tool, recommend_modulation_tool, evalualte_dab_tool, 
         simulation_verification_tool, pe_gpt_introduction_tool, train_pann_tool, 
         other_tasks_tool], llm=llm, verbose=True)
    
    return agent



