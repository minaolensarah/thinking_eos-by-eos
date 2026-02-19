# Diffusion LLMs can think eos-by-eos 

This repository contains code and datasets to reproduce the experiments from the paper "Diffusion LLMs can think eos-by-eos". 
<img src="figure1.png" alt="Illustration of Diffusion LLMs can think eos-by-eos" />


- A prompting experiment to compare autoregressive and diffusion LLMs without chain of thought on the tasks Addition, Entity Tracking, and Sudoku, where we find that diffusion LLMs outperform autoregressive ones given a sufficiently high generation length. They pad the additional space with EoS tokens.
Therefore, we hypothesize that diffusion LLMs use the EoS tokens as additional compute, which we name thinking EoS-by-EoS
- A controlled prompting experiment in which we keep the number of masks to predict constant and augment an increasing number of EoS tokens in the start state. We find that EoS tokens have a positive impact on the performance of LLaDA1.5 and Dream v0, but not on LLaDA2.0.
- Causal interventions that swap the hidden states of the EoS tokens between an original and a counterfactual run, which shows that they contain reasoning towards the final generation.
- A comparison between verbose CoT and thinking EoS-by-EoS regarding the trade-off between the token requirement and the accuracy gain.

## Setup
All required packages can be found in requirements.txt. You can install them with ```pip install -r requirements.txt```

## Usage
The folder ```src/data_generation``` contains the scripts to generate datapoints for the Addition and Entity Tracking tasks. The Sudokus can be generated with [Sudoku4LLM](https://github.com/DolbyUUU/Sudoku4LLM). Our exact datasets can be found in ```datasets```. Moreover, this repository contains code to pad EoS tokens into the starting state, to prompt the LLMs, and to perform activation patching, as well as the evaluation scripts. The file ```run_experiments.sh``` contains all commands to replicate the results for one model. 


## Citation
If you use this code in your work, please cite the paper:
```ToDo```

