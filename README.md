# PE-GPT: a New Paradigm for Power Electronics Design

[![DOI](https://img.shields.io/badge/DOI-10.1109/TIE.2024.3454408-cyan)](https://doi.org/10.1109/TIE.2024.3454408)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect--Fanfan%20Lin-blue)](https://www.linkedin.com/in/fanfanlin/)
[![ORCID](https://img.shields.io/badge/ORCID-Fanfan%20Lin-brightgreen)](https://orcid.org/0000-0002-5562-2478)
[![GitHub](https://img.shields.io/badge/Github-XinzeLee-black?logo=github)](https://github.com/XinzeLee)
[![IEEE](https://img.shields.io/badge/IEEE-Xplore-orange)](https://ieeexplore.ieee.org/document/10701612)
[![ResearchGate](https://img.shields.io/badge/ResearchGate-OpenAccess-blue)](https://www.researchgate.net/publication/384548400_PE-GPT_A_New_Paradigm_for_Power_Electronics_Design)

<!-- traffic:start -->
<p align="center">
  <a href="https://github.com/XinzeLee/PE-GPT/graphs/traffic">
    <img src="https://img.shields.io/badge/Total_Views-321-2563eb?style=flat-square" alt="Total repository views: 321" />
  </a>
  <a href="https://github.com/XinzeLee/PE-GPT/graphs/traffic">
    <img src="https://img.shields.io/badge/Total_Clones-137-7c3aed?style=flat-square" alt="Total repository clones: 137" />
  </a>
  <a href="https://github.com/XinzeLee/PE-GPT/graphs/traffic">
    <img src="https://img.shields.io/badge/Unique_Clones-87-b45309?style=flat-square" alt="Unique repository clones: 87" />
  </a>
</p>

<p align="center"><sub>Github traffic (monitoring started on May, 23, 2026) · cumulative tracked totals · Till 2026-06-22 UTC</sub></p>
<!-- traffic:end -->

## Description

We proposed PE-GPT in early 2024, **the first generative AI agent specifically designed for power electronics**, advancing AI-aided workflows toward an AI-native autonomous paradigm. Through collective efforts across the power electronics community, PE-GPT has been developed and validated across a broad range of design problems, including flyback, DAB, buck, and boost converter design; magnetics design including inductors and PCB transformers; and power semiconductor characterization and design.
<br><br>

<span style="font-size:20px;"><b>Keeping "AI-for-Good" as our mission, PE-GPT strives to revolutionize the paradigm for diverse power electronics tasks.</b></b></span>

For more information and collaborations, please visit the [official website of PE-GPT](https://fannie1803.github.io/pegpt.ai).


## The Proposed Hybrid Framework of PE-GPT
![The hybrid framework of PE-GPT.](https://github.com/user-attachments/assets/fa246d51-ea4e-4fce-967f-b584ee5da586)
<br>Fig. 1. The hybrid framework of PE-GPT.


## Demo Videos of PE-GPT
Demo videos of using PE-GPT for the power electronics design tasks.
<br>


**Demo Case 1:**
  * Flyback Converter Design with Component Selection

https://github.com/user-attachments/assets/5cd2e36d-b3b3-4ec4-a948-7841d27a756d

<br>

**Demo Case 2:**
  * Modulation Optimization for DAB Converters - 1

https://github.com/user-attachments/assets/53f07316-3a34-411e-86c7-fe621bb5a53c

<br>

**Demo Case 3:**
  * Modulation Optimization for DAB Converters - 2

https://github.com/user-attachments/assets/7532419a-2819-4fda-98c8-38dfe992708d

<br>

**Demo Case 4:**
  * Circuit Parameter Design for Buck Converters

https://github.com/user-attachments/assets/2e8ff52e-e2e1-41b5-9825-b0e65e2615c1


## Deploy PE-GPT on your PC
* To deploy PE-GPT on your PC, the first step is to setup your API call to OpenAI models, please see core/llm/llm.py for more details. <br>
* If you want to interact with Plecs software to simulate the designed modulation for DAB, you need to enable the xml-rpc interface in Plecs settings,
and to add the directory "core/simulation/devices" in the device library searching path in plecs.
<br><br>
```bash

# clone the github repository
git clone https://github.com/XinzeLee/PE-GPT

# change the current working directory
cd PE-GPT

# install all required dependencies
pip install -r requirements_specific.txt

# run the GUI and chat with PE-GPT
streamlit run main.py

```
<br><br>

### Install with Conda
To avoid version or env conflicts, you may use Conda for specific verions: Tested with python `3.10.18`

```bash
# clone the github repository
git clone https://github.com/XinzeLee/PE-GPT

# Navigate to the working directory
cd PE-GPT

# Create environment
conda env create -f environment.yml

# Activate the created environment
conda activate pe-gpt

streamlit run main.py
```

## Reference
@reference: Fanfan Lin, Xinze Li, Weihao Lei, Juan J. Rodriguez-Andina, Josep M. Guerrero, Changyun Wen, Xin Zhang, and Hao Ma, "PE-GPT: a New Paradigm for Power Electronics Design", IEEE Transactions on Industrial Electronics.
<br>
@code-author: Xinze Li (email: xinzeli831@gmail.com), Fanfan Lin (email: fanfanlin31@gmail.com)

<br><br>
## Notes
* This repository provides a simplified version of the PE-GPT methodology presented in our journal paper. Despite the simplifications, the released code preserves the overall core architecture of the proposed PE-GPT.
<br><br>
* This repository currently includes the following functions/blocks: Retrieval augmented generation, LLM agents, Model Zoo (with a physics-in-architecture neural network, PANN, for modeling DAB converters), metaheuristic algorithm for optimization, simulation verification, graphical user interface, and knowledge base. Please note that the current knowledge base is a simplified version for illustration. 

<br><br>
## License

This project is licensed under the GNU AGPLv3 with additional terms restricting commercial training.

While the code may be used freely in open source projects and in academic teaching/research, it may not be used to provide paid
courses or commercial training services. See the [GNU AGPLv3](./LICENSE) file for full terms.

