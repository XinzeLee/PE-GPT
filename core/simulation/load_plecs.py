# This file is used to interface plecs models
"""
@reference: PE-GPT: a New Paradigm for Power Electronics Design, by Fanfan Lin, Xinze Li, et al.
@code-author: Xinze Li, Fanfan Lin, Weihao Lei
@github: https://github.com/XinzeLee/PE-GPT
"""

import os
import threading
import xmlrpc.client
import subprocess
import platform
import streamlit as st

from func_timeout import func_set_timeout




class PlecsThread(threading.Thread):

    _opts = {'ModelVars': {'Tstop':0.1, 'Tdead':200e-9, 'Vin': 200, 
                          'Vref': 200, 'P': 800, 'Ro': 800 ** 2 / 80}}

    def __init__(self, model_name, file_path, **kwargs):
        super(PlecsThread, self).__init__()
        self.model_name = model_name
        self.model_path = file_path
        self.server = xmlrpc.client.Server('http://localhost:1080/RPC2"')
        self.kwargs = kwargs
        
    def update_opts(self):
        opts = {'ModelVars': {'Tstop': self.__class__._opts['ModelVars']['Tstop'],
                              'Tdead': self.__class__._opts['ModelVars']['Tdead']}}
        opts['ModelVars'].update(self.kwargs)
        return opts

    def load(self):
        self.server.plecs.load(self.model_path)

    def close(self):
        self.server.plecs.close(self.model_name)

    @func_set_timeout(30)
    def run_once(self, opts):
        self.load()
        self.server.plecs.simulate(self.model_name, opts)
        # self.close() # if you want to close it automatically

    def run(self):
        try:
            opts = self.update_opts()
            self.run_once(opts)
        except Exception as e:
            st.write(e)


def open_plecs(file_path):
    system = platform.system()
    if system == 'Windows':
        os.startfile(file_path)
    elif system == 'Darwin':  # macOS
        subprocess.call(["open", file_path])
    elif system == 'Linux':
        subprocess.call(["xdg-open", file_path])


def dab_plecs(modulation, Vin, Vref, P, *mod_params):
    from ..model_zoo.pann_dab_vars import n
    # load the plecs file
    model_name = "DAB"
    file_path = os.path.abspath(f"core/simulation/{model_name}.plecs")
    open_plecs(file_path)
    
    mod_params = [float(item) for item in mod_params]
    st.write(f"{modulation}")
    # load the operating conditions and modulation parameters
    if modulation == "SPS":
        D1, D2, phi1, phi2 = 1., 1., 0., 0.
    elif modulation == "DPS":
        D1, D2, phi1, phi2 = mod_params[0], mod_params[0], 0., 0.
    elif modulation == "EPS":
        if n*Vref < Vin: D1, D2, phi1, phi2 = mod_params[0], 1, 0., 0.
        else: D1, D2, phi1, phi2 = 1, mod_params[0], 0., 0.
    elif modulation == "TPS":
        D1, D2, phi1, phi2 = *mod_params, 0., 0.
    elif modulation == "5DOF":
        D1, D2, phi1, phi2 = mod_params
    kwargs = {"Vin": Vin, "Vref": Vref, "P": P, "D1": D1, 
              "D2": D2, "phi1": phi1, "phi2": phi2, "Ro": Vref**2/P}
    
    # conduct the plecs simulation
    thread = PlecsThread(model_name, file_path, **kwargs)
    thread.start()
