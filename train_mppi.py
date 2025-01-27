import sys
import logging
import os
import shutil
import time
from collections import deque
from rl import utils
import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import gym

from rl.ppo import PPO
from rl.vec_env.envs import make_vec_envs
from rl.model import Policy
from rl.pas_rnn_model import Label_VAE, Sensor_VAE
from rl.storage import RolloutStorage
from evaluation_mppi import evaluate as evaluate_mppi
from evaluation import evaluate
from mppi.mppi import MPPI_Planner
from test import test
from crowd_sim import *


import warnings
warnings.filterwarnings("ignore")
import pdb

## TODO: 
# 1. Check the infer
# 2. Check the rollouts
# 3. Check the learning rate scheduler
# 4. Update hyperparams

def main():

	from arguments import get_args
	algo_args = get_args()

	# save policy to output_dir
	if os.path.exists(algo_args.output_dir) and algo_args.overwrite: # if I want to overwrite the directory
		shutil.rmtree(algo_args.output_dir)  # delete an entire directory tree

	if not os.path.exists(algo_args.output_dir):
		os.makedirs(algo_args.output_dir)

		shutil.copytree('crowd_nav/configs', os.path.join(algo_args.output_dir, 'configs'))
		shutil.copy('arguments.py', algo_args.output_dir)
	from crowd_nav.configs.config import Config
	config = Config()


	# configure logging
	log_file = os.path.join(algo_args.output_dir, 'output.log')
	mode = 'a' # if algo_args.resume else 'w'
	file_handler = logging.FileHandler(log_file, mode=mode)
	stdout_handler = logging.StreamHandler(sys.stdout)
	level = logging.INFO
	for handler in logging.root.handlers[:]:
		logging.root.removeHandler(handler)
	logging.basicConfig(level=level, handlers=[stdout_handler, file_handler],
						format='%(asctime)s, %(levelname)s: %(message)s', datefmt="%Y-%m-%d %H:%M:%S")

	
	torch.manual_seed(algo_args.seed)
	torch.cuda.manual_seed_all(algo_args.seed)
     
	if algo_args.cuda:
		if algo_args.cuda_deterministic:
			# reproducible but slower
			torch.backends.cudnn.benchmark = False
			torch.backends.cudnn.deterministic = True
		else:
			# not reproducible but faster
			torch.backends.cudnn.benchmark = True
			torch.backends.cudnn.deterministic = False



	torch.set_num_threads(algo_args.num_threads)
	device = torch.device("cuda" if algo_args.cuda else "cpu")



	summary_path = algo_args.output_dir+'/runs_gradient'
	if not os.path.exists(summary_path):
		os.makedirs(summary_path)


	# For fastest training: use GRU
	env_name = algo_args.env_name
	recurrent_cell = 'GRU'

	# Create a wrapped, monitored VecEnv
	if config.robot.policy == 'pas_mppi': # using only one process for mppi
		envs = make_vec_envs(env_name, algo_args.seed, 1, 
							algo_args.gamma, device, False, envConfig=config, phase='train')
	else:
		envs = make_vec_envs(env_name, algo_args.seed, algo_args.num_processes,
							algo_args.gamma, device, False, envConfig=config, phase='train')
	val_envs = make_vec_envs(env_name, algo_args.seed, 1,
						 algo_args.gamma, device, allow_early_resets=True,
						 envConfig=config, phase='val')
		
	#################################
	#      PREPARE MODEL   #
	################################# 
	dummy_high = np.inf * np.ones([2, ])
	dummy_action_space = gym.spaces.Box(-dummy_high, dummy_high, dtype=np.float32)
	# actor_critic = Policy(
	# 	dummy_action_space,
	# 	config = config,
	# 	base_kwargs=algo_args,
	# 	base=config.robot.policy)
 
	# actor_critic.base.nenv = 1 # TODO: use multi envs
 
	# Load PaS model 
	label_ckpt_dir = 'data/crossing_H12/label_vae_ckpt/label_weight_300.pth'
	label_vae = Label_VAE(algo_args).to(device)
	label_vae.load_state_dict(torch.load(label_ckpt_dir))
	pas_ckpt_dir = 'data/crossing_H12/sensor_vae_woEstLoss_ckpt/sensor_weight_300.pth'
	sensor_vae = Sensor_VAE(algo_args, config).to(device)
	sensor_vae.load_state_dict(torch.load(pas_ckpt_dir))
 
	mppi_module = MPPI_Planner(config, algo_args, device)
 
	# pas_ckpt_dir = 'data/pasrl_CircleFOV30_seed10/checkpoints'
	# load_path=os.path.join(pas_ckpt_dir, '38800.pt')
 
	# if os.path.exists(load_path):
	# 	actor_critic.load_state_dict(torch.load(load_path), strict=False)
	# 	actor_critic.base.nenv = 1
	# 	actor_critic.config = config
 

	if config.robot.policy=='pas_rnn' or config.robot.policy=='pas_diffstack' or config.robot.policy=='pas_mppi':
		rollouts = RolloutStorage(algo_args.num_steps,
								algo_args.num_processes,
								envs.observation_space.spaces,
								envs.action_space,
								algo_args.rnn_hidden_size,
								recurrent_cell_type=recurrent_cell,
								base=config.robot.policy, encoder_type=config.pas.encoder_type, seq_length=algo_args.seq_length, gridsensor=config.pas.gridsensor)
		eval_rollouts = RolloutStorage(int(config.env.time_limit/config.env.time_step),
								1,
								envs.observation_space.spaces,
								envs.action_space,
								algo_args.rnn_hidden_size,
								recurrent_cell_type=recurrent_cell,
								base=config.robot.policy, encoder_type=config.pas.encoder_type, seq_length=algo_args.seq_length, gridsensor=config.pas.gridsensor)

	# allow the usage of multiple GPUs to increase the number of examples processed simultaneously
	# nn.DataParallel(actor_critic).to(device)
 
	# agent = PPO(
	# 	actor_critic,
	# 	algo_args.clip_param,
	# 	algo_args.ppo_epoch,
	# 	algo_args.num_mini_batch,
	# 	algo_args.value_loss_coef,
	# 	algo_args.entropy_coef,
	# 	PaS_coef = config.pas.PaS_coef,
	# 	lr=algo_args.lr,
	# 	eps=algo_args.eps,
	# 	max_grad_norm=algo_args.max_grad_norm)
 
	obs = envs.reset()
	if isinstance(obs, dict):
		for key in obs:
			rollouts.obs[key][0].copy_(obs[key])
	else:
		rollouts.obs[0].copy_(obs)

	rollouts.to(device)


	recurrent_hidden_states = {}
	for key in rollouts.recurrent_hidden_states:
		recurrent_hidden_states[key] = rollouts.recurrent_hidden_states[key][0]


	episode_rewards = deque(maxlen=100)

	start = time.time()
	num_updates = int(
		algo_args.num_env_steps) // algo_args.num_steps // algo_args.num_processes


	for j in range(num_updates): 
		## Validation for the saving intervals
		total_num_steps = (j + 1) * algo_args.num_processes * algo_args.num_steps
		visualize = True
		_, _ = evaluate_mppi(eval_rollouts, config, algo_args.output_dir, label_vae, sensor_vae, mppi_module, val_envs, device, config.env.val_size, logging, visualize, 'val', j)

		break
  

if __name__ == '__main__':
	main()
