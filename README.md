# PaS_Disambig

## Setup
0. Requirements: python3.8 (Tested on Ubuntu 18.04)
1. Install crowd_sim and crowd_nav into pip
```
pip install -e .
pip install -r requirements.txt
```
2. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
</p>

## Getting started
This repository is organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains code for training and testing the policies. The folder rl contains the code
for the network and PPO algorithm.
Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies.

### Change configurations
1. Environment configurations: modify `crowd_nav/configs/config.py` for vae_pretrain.py (PaS Inference Training) and `crowd_nav/configs/config_mppi.py` for running train_mppi.py.
- For perception level (ground-truth or sensor): set `pas.gridsensor` to `gt` or `sensor`.
- For PaS perception with occlusion inference: set `pas.encoder_type` to `vae` (otherwise `cnn`).
- For a sequential grid (or single) input  : sets `pas.seq_flag` to `True` (or `False`).

2. MPPI/PPO configurations: modify arguments.py 

### Run the code

1. Collect data for training GT-VAE.
- In `crowd_nav/configs/config.py`, set (i) `robot.policy` to `orca` and (ii) `sim.collectingdata` to `True`
- In `arguments.py`, set `output_dir` to `VAEdata_CircleFOV30/{phase}` where phase is `train` or `val` or `test`
Run the following commands for all three phases.
```
python collect_data.py 
```
2. Pretrain GT-VAE (label_vae) and PaS-VAE (sensor_vae) with collected data. Update the `output_path` to data path in `vae_pretrain.py`
```
python vae_pretrain.py 
```
3. Run mppi with pretained vae. There is no actual training involved in this step, but evaluation.
- In `train_mppi.py`, modify `label_ckpt_dir` and `pas_ckpt_dir`
- In `arguments.py`, set `output_dir` to `data/{foldername}` (i.e. `data/mppi`)
```
python train_mppi.py 
```


## Citation
If you find the code or the paper useful for your research, please cite our paper:
```
Y.-J. Mun, M. Itkina, S. Liu, and K. Driggs-Campbell. "Occlusion-Aware Crowd Navigation Using People as Sensors". ArXiv, 2022.
```

## Credits
Part of the code is based on the following repositories:  
[1] S. Liu, P. Chang, W. Liang, N. Chakraborty, and K. Driggs-Campbell, “Decentralized structural-RNN for robot crowd navigation 
with deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), IEEE, 2021.
(Github:https://github.com/Shuijing725/CrowdNav_Prediction)

[2] C. Chen, Y. Liu, S. Kreiss, and A. Alahi, “Crowd-robot interaction: Crowd-aware robot navigation with attention-based deep reinforcement learning,” in International Conference on Robotics and Automation (ICRA), 2019, pp. 6015–6022.
(Github: https://github.com/vita-epfl/CrowdNav)

[3] I. Kostrikov, “Pytorch implementations of reinforcement learning algorithms,” https://github.com/ikostrikov/pytorch-a2c-ppo-acktr-gail, 2018.

