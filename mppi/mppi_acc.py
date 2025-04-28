# This code is from https://github.com/UM-ARM-Lab/pytorch_mppi/blob/master/src/pytorch_mppi/mppi.py
# pip install arm-pytorch-utilities
import os
import logging
import time
import typing
import torch
from torch.distributions.multivariate_normal import MultivariateNormal
from functorch import vmap
from matplotlib import pyplot as plt
from arm_pytorch_utilities import handle_batch_input
from mppi.utils import * # compute_disambig_cost
from copy import deepcopy
import pdb

logger = logging.getLogger(__name__)


def _ensure_non_zero(cost, beta, factor):
    return torch.exp(-factor * (cost - beta))


# TODO: improve transfer_to_ego_grid to be more efficient and work with various batch size in either direction. Maybe try using https://github.com/getkeops/keops/tree/main.
# TODO: use the PaS next sensor grid to compute the costs
# TODO: check the Disambig formulation


# 4/3
# TODOV: 1. Add wall information to the MPPI function


# 4/8
# TODO: Make robot movement larger.


# Set random seed for reproducibility
seed = 0
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
# If using other libraries for randomness
import random
import numpy as np
random.seed(seed)
np.random.seed(seed)


class SpecificActionSampler:
    def __init__(self):
        self.start_idx = 0
        self.end_idx = 0
        self.slice = slice(0, 0)

    def sample_trajectories(self, state, info):
        raise NotImplementedError

    def specific_dynamics(self, next_state, state, action, t):
        """Handle dynamics in a specific way for the specific action sampler; defaults to using default dynamics"""
        return next_state

    def register_sample_start_end(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.slice = slice(start_idx, end_idx)
        
class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, kinematics, running_cost, nx, noise_sigma, num_samples=100, horizon=30, device="cpu",
                 terminal_state_cost=None,
                 lambda_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 rollout_samples=1,
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 sample_null_action=False,
                 specific_action_sampler: typing.Optional[SpecificActionSampler] = None,
                 noise_abs_cost=False):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param noise_abs_cost: Whether to use the absolute value of the action noise to avoid bias when all states have the same cost
        """
        self.d = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        # handle 1D edge case
        if self.nu == 1:
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.d)
            self.u_max = self.u_max.to(device=self.d)

        self.noise_mu = noise_mu.to(self.d)
        self.noise_sigma = noise_sigma.to(self.d)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.d)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))

        self.step_dependency = step_dependent_dynamics # not used
        self.F = dynamics
        self.kinematics = kinematics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.specific_action_sampler = specific_action_sampler
        self.noise_abs_cost = noise_abs_cost
        self.state = None
        self.info = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.M = rollout_samples
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None
        self.curr_map_xy = None
        self.curr_label_grid = None
        self.curr_sensor_grid = None
        self.next_map_xy = None
        self.next_sensor_grid = None

    # @handle_batch_input(n=2)
    def _dynamics(self, state, u):
        return self.F(state, u) 

    # @handle_batch_input(n=2)
    def _running_cost(self, state, ob, next_state, decoded, wall_polygons, t):
        return self.running_cost(state, ob, next_state, decoded, wall_polygons, t)# if self.step_dependency else self.running_cost(state, u)

    def shift_nominal_trajectory(self):
        """
        Shift the nominal trajectory forward one step
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def command(self, state, ob, decoded, wall_polygons, shift_nominal_trajectory=False, info=None):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
        if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :returns action: (nu) best action
        """
        self.info = info
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()
        return self._command(state, ob, decoded, wall_polygons)
    
    def _compute_weighting(self, cost_total):
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        return self.omega
    
    def _command(self, state, ob, decoded, wall_polygons):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        self.ob = ob.to(dtype=self.dtype, device=self.d)
        self.decoded = decoded.to(dtype=self.dtype, device=self.d)
        self.wall_polygons = [wall.to(dtype=self.dtype, device=self.d) for wall in wall_polygons]
        cost_total = self._compute_total_cost_batch()
        self._compute_weighting(cost_total)
        perturbations = torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)

        self.U = self.U + perturbations
        action = self.U[:self.u_per_command]

        # # reduce dimensionality if we only need the first command
        # if self.u_per_command == 1:
        #     action = action[0]
        # return action
            
        propagated_states = []
        ## Rollouts with the final action
        for a in range(self.u_per_command):
            if self.kinematics == "holonomic":
                state = self._dynamics(state[:2], action[a])
                propagated_states.append(state)
            elif self.kinematics == "unicycle":
                state = self._dynamics(state[:4], action[a])
                propagated_states.append(state)
            
        propagated_states = torch.stack(propagated_states)
        return propagated_states, action

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            self.U = torch.cat((self.U, self.u_init.repeat(horizon - self.U.shape[0], 1)))
        self.T = horizon

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = self.noise_dist.sample((self.T,))

    def _compute_rollout_costs(self, perturbed_actions):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.d, dtype=self.dtype)
        cost_samples = cost_total.repeat(self.M, 1)
        cost_var = torch.zeros_like(cost_total)

        state = self.state.view(1, -1).repeat(K, 1)

        # rollout action trajectory M times to estimate expected cost
        # (batch, K, 2) (1, 100, 2)
        if self.kinematics == "holonomic":
            state = state.repeat(self.M, 1, 1)[:, :, :2] # repeated current state # TODO: not using self.M
        elif self.kinematics == "unicycle":
            state = state.repeat(self.M, 1, 1)[:, :, :4] # px, py, theta, v 

        
        # (batch, K, pred_horizon, human_num, 7) # repeated obs with pred_horizon
        ob = self.ob.unsqueeze(0).unsqueeze(0).repeat(self.M, K, 1, 1, 1) # vector obs

                
        states = []
        actions = []
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            next_state = self._dynamics(state, u) # next state is compuated in running_cost to compute cost.
            # next_state = self._sample_specific_dynamics(next_state, state, u, t)
            # state = next_state
            c = self._running_cost(state, ob, next_state, self.decoded, self.wall_polygons, t) 
            # next_state = self._dynamics(state, u)
            # next_state = self._sample_specific_dynamics(next_state, state, u, t)
            state = next_state # self._dynamics(state, u) 
            # state = self._dynamics(state, u)
            cost_samples = cost_samples + c / (t+1) #* 0.8**t
            if self.M > 1:
                cost_var += c.var(dim=0) * (self.rollout_var_discount ** t)

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # Actions is K x T x nu
        # States is K x T x nx
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # action perturbation cost
        if self.terminal_state_cost:
            c = self.terminal_state_cost(states, actions)
            cost_samples = cost_samples + c
        cost_total = cost_total + cost_samples.mean(dim=0)
        cost_total = cost_total + cost_var * self.rollout_var_cost
        return cost_total, states, actions
    
    def _compute_perturbed_action_and_noise(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + noise
        perturbed_action = self._sample_specific_actions(perturbed_action)
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U
        
    def _sample_specific_actions(self, perturbed_action):
        # specific sampling of actions (encoding trajectory prior and domain knowledge to create biases)
        i = 0
        if self.sample_null_action:
            perturbed_action[i] = 0
            i += 1
        if self.specific_action_sampler is not None:
            actions = self.specific_action_sampler.sample_trajectories(self.state, self.info)
            # check how long it is
            actions = actions.reshape(-1, self.T, self.nu)
            perturbed_action[i:i + actions.shape[0]] = actions
            self.specific_action_sampler.register_sample_start_end(i, i + actions.shape[0])
            i += actions.shape[0]
        return perturbed_action
    
    def _sample_specific_dynamics(self, next_state, state, u, t):
        if self.specific_action_sampler is not None:
            next_state = self.specific_action_sampler.specific_dynamics(next_state, state, u, t)
        return next_state

    def _compute_total_cost_batch(self):
        self._compute_perturbed_action_and_noise()
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv  # Like original paper

        rollout_cost, self.states, actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions = actions / self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost
        return self.cost_total

    # TODO: might need to update this bound to be differentiable later
    def _bound_action(self, action):
        bounded_action = action.clone()
        if self.u_max is not None:
            for t in range(self.T):
                # TODO: does action needs to be flattened? Then, slice_control can be brought back.
                u = action[:, t].clone() # self._slice_control(t)]
                # cu = torch.max(torch.min(u, self.u_max), self.u_min)
                if self.kinematics == "holonomic":
                    v_norm = torch.linalg.norm(u, axis=-1)
                    cu = u.clone()
                    # pdb.set_trace()
                    # if v_norm > self.u_max:
                        # print(action.shape)
                    ind = torch.where(v_norm > self.u_max)[0] # for holonomic dynamics
                    # TODO: implement for unicycle dynamics
                    cu[ind,0] = u[ind,0] / v_norm[ind] * self.u_max
                    cu[ind,1] = u[ind,1] / v_norm[ind] * self.u_max
                elif self.kinematics == "unicycle":
                    cu = u.clone()  # acc.py: bound action should be done also in dynamic
                    cu[:,0] = torch.clip(u[:,0], self.u_min[0], self.u_max[0])
                    cu[:,1] = torch.clip(u[:,1], self.u_min[1], self.u_max[1])
                    # cu[:,0] = torch.max(torch.min(u[:,0], self.u_max[0]), self.u_min[0])
                    # cu[:,1] = torch.max(torch.min(u[:,1], self.u_max[1]), self.u_min[1])
                bounded_action[:, t]=cu.clone() # self._slice_control(t)] = cu
                if self.sample_null_action:
                    bounded_action[self.K - 1] = 0
        return bounded_action

    def _slice_control(self, t):
        return slice(t * self.nu, (t + 1) * self.nu)

    def get_rollouts(self, state, num_rollouts=1):
        """
            :param state: either (nx) vector or (num_rollouts x nx) for sampled initial states
            :param num_rollouts: Number of rollouts with same action sequence - for generating samples with stochastic
                                 dynamics
            :returns states: num_rollouts x T x nx vector of trajectories

        """
        state = state.view(-1, self.nx)
        if state.size(0) == 1:
            state = state.repeat(num_rollouts, 1)

        T = self.U.shape[0]
        states = torch.zeros((num_rollouts, T + 1, self.nx), dtype=self.U.dtype, device=self.U.device)
        states[:, 0] = state
        for t in range(T):
            states[:, t + 1] = self._dynamics(states[:, t].view(num_rollouts, -1),
                                              self.u_scale * self.U[t].tile(num_rollouts))
        return states[:, 1:]



class MPPI_Planner():
    def __init__(self,config, args, device):
        self.config = config
        self.args = args
        self.device = device
        self.sequence = self.config.pas.sequence
        self.lookahead_steps = self.config.diffstack.lookahead_steps
        self.full_sequence = self.sequence + self.lookahead_steps
        self.human_num = self.config.sim.human_num
        self.time_step = self.config.env.time_step
        
        self.curr_label_grid = None
        self.curr_sensor_grid = None
        self.curr_map_xy = None
        self.vis_ids = None
        self.disambig_R_map = None
        self.disambig_W_map = None
        self.next_map_xy = None
        self.next_sensor_grid = None
        
        self._init_mppi()
        self.step = 0
        
    def _init_mppi(self):
        nx = 2
        noise_sigma = torch.diag(torch.rand(nx, dtype=torch.double)*2) # torch.eye(nx, device=self.device, dtype=torch.double)
        N_SAMPLES = self.config.diffstack.num_samples
        TIMESTEPS = self.lookahead_steps
        lambda_ = self.config.diffstack.lambda_
        self.kinematics = self.config.action_space.kinematics
        if self.kinematics == "holonomic":
            ACTION_LOW = -self.config.robot.v_pref # Not used
            ACTION_HIGH = self.config.robot.v_pref
        elif self.kinematics == "unicycle":
            ACTION_LOW = torch.tensor([0., -np.pi*self.config.robot.w_max], dtype=torch.double, device=self.device)
            ACTION_HIGH = torch.tensor([self.config.robot.v_pref, self.config.robot.w_max], dtype=torch.double, device=self.device)
            self.v_pref = self.config.robot.v_pref
        else:
            raise ValueError("Invalid kinematics")
        
        
        self.mppi_gym = MPPI(self.dynamics, self.kinematics, self.running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, device=self.device, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=self.device),
                         u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=self.device), u_per_command=self.lookahead_steps,
                        step_dependent_dynamics=False, sample_null_action=True)
        
        # self.mppi_gym = KMPPI(self.dynamics, self.running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
        #                     lambda_=lambda_, device=self.device, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=self.device),
        #                     u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=self.device), u_per_command=self.lookahead_steps,
        #                     step_dependent_dynamics=False, sample_null_action=True)
        
    def dynamics(self, state, action):
        """
        state: robot's current state (K,2)
        """
        # print(state.shape, action.shape)
        
        # State only containes robot's position
        if self.kinematics == "holonomic":
            next_r_state = state + action * self.time_step
        elif self.kinematics == "unicycle":
            #### acc.py#############
            theta = state[...,2] 
            v = torch.clip(state[...,3] + action[...,0] * self.time_step, 0., self.v_pref)  # v' = v + a*dt     
            # differential drive            
            w = action[...,1]/self.time_step # action.r is delta theta
            R = action[...,0]/w

            next_x = state[...,0] - R * torch.sin(theta) + R * torch.sin(theta + action[...,1])
            next_y = state[...,1] + R * torch.cos(theta) - R * torch.cos(theta + action[...,1])
            
            epsilon = 0.0001       
            mask = torch.abs(action[...,1]) < epsilon

            next_x[mask] = state[mask][...,0] + v[mask] * self.time_step * torch.cos(theta[mask])
            next_y[mask] = state[mask][...,1] + v[mask] * self.time_step * torch.sin(theta[mask])
                
            next_theta = (theta+ action[...,1]) % (2 * np.pi) # action[...,1]: angular change, not angular velocity (w).
            next_r_state = torch.stack([next_x, next_y, next_theta, v], dim=-1)
            # pdb.set_trace()
            ######################
        else:
            raise ValueError("Invalid kinematics")        
        return next_r_state
    
    def visualize_traj(self,r_state, next_r_state, goal_pos, highlight_indices=None):
        """
        r_state: robot's current state (K,2)
        next_r_state: robot's next state (K,2)
        """
        ## Sanity check: Visualize the human and robot's positions in circle and masks in multiple plots
        # make visualizaton directory
        save_dir = "mppi_sanitycheck"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # count how many is already saved
        k = len(os.listdir(save_dir)) 
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        ax.tick_params(labelsize=16)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        ax.plot(r_state[0,:,0].cpu(), r_state[0,:,1].cpu(), 'b*', markersize=10)
        ax.plot(next_r_state[0,:,0].cpu(), next_r_state[0,:,1].cpu(), 'r+', markersize=10)
        ax.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'go', markersize=10)
        ax.set_title("Robot's current and sampled next state")
        ax.legend(["Current state", "Next state", "Goal"])
        
        # if highlight_indices is not None:
        #     # for idx in  highlight_indices:
        #     ax.plot(r_state[0,highlight_indices,0].cpu(), r_state[0,highlight_indices,1].cpu(), 'bo', markersize=10)
        #     ax.plot(next_r_state[0,highlight_indices,0].cpu(), next_r_state[0,highlight_indices,1].cpu(), 'ro', markersize=10)
        
        # for ax, grid_info in zip(axes.flatten(), vis_grids):
        #     # use contourf to create a filled contour plot
        #     map_x, map_y, grid, title = grid_info
        #     ax.tick_params(labelsize=16)
        #     ax.set_xlim(-6, 6)
        #     ax.set_ylim(-6, 6)
        #     ax.set_xlabel('x(m)', fontsize=16)
        #     ax.set_ylabel('y(m)', fontsize=16)
        #     if "id" in title:
        #         CS = ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='tab20', levels=np.linspace(0, 6, 6))
        #         # add the color bar
        #         cbar = plt.colorbar(CS)
        #     else:
        #         ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='binary', levels=np.linspace(0, 1, 9))
        #     ax.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'ro', markersize=10)
        #     ax.set_title(title)

        # save the figure
        # plt.title("Visible ids: " + str(self.vis_ids))
        plt.savefig(save_dir+"/vis_traj_" + str(k)+".png")
        plt.close()
        

    def visualize_disambig_traj(self,r_state, next_r_state, goal_pos, highlight_indices=None, map_xy=None, sensor_grid=None, wall_polygons=None, h_states=None, lowest_cost_ind=None, cost=None):
        """
        r_state: robot's current state (K,2)
        next_r_state: robot's next state (K,2)
        """
        ## Sanity check: Visualize the human and robot's positions in circle and masks in multiple plots
        # make visualizaton directory
        save_dir = "mppi_sanitycheck"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # count how many is already saved
        k = len(os.listdir(save_dir)) 
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        ax.tick_params(labelsize=16)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        ax.plot(r_state[0,:,0].cpu(), r_state[0,:,1].cpu(), 'b*', markersize=10)
        ax.plot(next_r_state[0,:,0].cpu(), next_r_state[0,:,1].cpu(), 'b+', markersize=10)
        ax.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'y*', markersize=10)
        ax.set_title("Cost:{}".format(cost))
        
        if highlight_indices is not None:
            # for idx in  highlight_indices:
            ax.plot(r_state[0,highlight_indices,0].cpu(), r_state[0,highlight_indices,1].cpu(), 'bo', markersize=10)
            ax.plot(next_r_state[0,highlight_indices,0].cpu(), next_r_state[0,highlight_indices,1].cpu(), 'go', markersize=10)
    
        if lowest_cost_ind is not None:
            ax.plot(r_state[0,lowest_cost_ind,0].cpu(), r_state[0,lowest_cost_ind,1].cpu(), 'r*', markersize=10)
            ax.plot(next_r_state[0,lowest_cost_ind,0].cpu(), next_r_state[0,lowest_cost_ind,1].cpu(), 'ro', markersize=10)
        
        if sensor_grid is not None:
            map_x, map_y = map_xy
            ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), sensor_grid.cpu().numpy(), cmap='binary', levels=np.linspace(0, 1, 9))
            
            
        ## add humans
        for h_state in h_states:
            ax.plot(h_state[:,0].cpu(), h_state[:,1].cpu(), 'g*', markersize=10)       
            
        ax.legend(["Current state", "Next state", "Goal", "Disambig curr state", "Disambig next state", "Lowest cost current state", "Lowest cost next state"]) 
        
        
        ## add walls
        if wall_polygons is not None and len(wall_polygons) > 0:
            for wall in wall_polygons:
                xmin, xmax, ymin, ymax = wall[:,0].min(), wall[:,0].max(), wall[:,1].min(), wall[:,1].max()
                width = xmax - xmin
                height = ymax - ymin
                
                wall_rec = plt.Rectangle((xmin.cpu().numpy(), ymin.cpu().numpy()), width.cpu().numpy(), height.cpu().numpy(), fill=False, color='grey', linestyle='--')
                ax.add_artist(wall_rec)
            
        # for ax, grid_info in zip(axes.flatten(), vis_grids):
        #     # use contourf to create a filled contour plot
        #     map_x, map_y, grid, title = grid_info
        #     ax.tick_params(labelsize=16)
        #     ax.set_xlim(-6, 6)
        #     ax.set_ylim(-6, 6)
        #     ax.set_xlabel('x(m)', fontsize=16)
        #     ax.set_ylabel('y(m)', fontsize=16)
        #     if "id" in title:
        #         CS = ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='tab20', levels=np.linspace(0, 6, 6))
        #         # add the color bar
        #         cbar = plt.colorbar(CS)
        #     else:
        #         ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='binary', levels=np.linspace(0, 1, 9))
        #     ax.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'ro', markersize=10)
        #     ax.set_title(title)

        # save the figure
        # plt.title("Visible ids: " + str(self.vis_ids))
        
        plt.savefig(save_dir+"/vis_traj_" + str(k)+".png")
        plt.close()
        

        

    
    def visualize(self,vis_grids, goal_pos, disambig_c):
        """
        vis_grids: list of lists of [map_xy, grid, title]
        goal_pos: robot's goal position
        """
        ## Sanity check: Visualize the human and robot's positions in circle and masks in multiple plots
        # make visualizaton directory
        if not os.path.exists("mppi_sanitycheck"):
            os.makedirs("mppi_sanitycheck")
        # count how many is already saved
        k = len(os.listdir("mppi_sanitycheck")) 
        
        fig, axes = plt.subplots(2, 5, figsize=(25, 10))
        for ax, grid_info in zip(axes.flatten(), vis_grids):
            # use contourf to create a filled contour plot
            map_x, map_y, grid, title = grid_info
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            if "id grid" in title:
                CS = ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='tab20', levels=np.linspace(0, 6, 6))
                # add the color bar
                cbar = plt.colorbar(CS)
            else:
                ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='binary', levels=np.linspace(0, 1, 9))
            ax.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'ro', markersize=10)
            ax.set_title(title)

        # save the figure
        plt.title("Visible ids: " + str(self.vis_ids))
        plt.savefig("mppi_sanitycheck/vis_" + str(k)+".png")
        plt.close()       
        

    def running_cost(self, state, ob, next_state, decoded, wall_polygons, t):
        """
        state: robot's state at t (batch, K, 2). Note that this is not the current state but a rolled state.
        ob: human's state at curr_t:curr_t+look_ahead (batch, K, 1+lookahead_steps, human_num, 7) 
        """
        if self.kinematics == "holonomic":
            r_state = state
            next_r_state = next_state # self.dynamics(state, action)#.unsqueeze(-2)
        elif self.kinematics == "unicycle":
            r_state = state[:,:,:2]
            next_r_state = next_state[:,:,:2] # (batch,K,2)
        r_radius = torch.tensor(self.config.robot.radius, device=self.device).repeat(r_state.shape[0],r_state.shape[1]) # (batch, K)
        r_goal = torch.tensor(self.r_goal, device=self.device).repeat(r_state.shape[0],r_state.shape[1],1) # (batch, K)
    
        # Generate label grid
        FOV_radius = self.config.robot.FOV_radius
        grid_res = self.config.pas.grid_res
        discomfort_dist = self.config.reward.discomfort_dist
        discomfort_penalty_factor = self.config.reward.discomfort_penalty_factor
        
        ########## Collision costs based on ego's state vector##################
        next_h_state = ob[:, :, t+1, :, :2] # (batch, K, human_num, 2)
        h_radius = ob[:, :, t+1, :, -1] # (batch, K, human_num, 1)
        collision_penalty = self.config.reward.collision_penalty
        
        repeated_next_r_state = next_r_state.unsqueeze(-2).repeat(1,1,self.human_num,1)
        
        ######## acc.py ########
        # Make invisible humans a far away dummy value
        decoded_in_unknown = torch.where(self.curr_sensor_grid==0.5, decoded[0,0], torch.tensor(0.0))
        decoded_human_id = torch.unique(self.curr_label_grid[1][decoded_in_unknown>0.4,])
        
        next_vis_h_state = next_h_state.clone()
        if len(decoded_human_id)>0:
            if len(decoded_human_id)>1: # For debugging in a static human behind wall scenario
                print(decoded_human_id)
                pdb.set_trace()
            for id in decoded_human_id:
                next_vis_h_state[:,:,int(id)] = torch.tensor([100,100]) # dummy position to invisible human
        
        dist = torch.linalg.norm(next_vis_h_state - repeated_next_r_state, dim=-1) # (batch, K, human_num)
        ################
        
        dmin = torch.min(dist, axis=-1)#-r_radius-h_radius[:, torch.argmin(dist, axis=-1)] # (batch, K)
        
        # Only consider the closest human
        closest_dist = dmin.values -r_radius-h_radius[torch.arange(h_radius.shape[0]), dmin.indices].squeeze(-1) # (batch, K)
    

        
        # Compute the collision cost for each (batch, K)
        collision_c = torch.where(closest_dist<0.,
                          torch.tensor(collision_penalty),
                          torch.tensor(0))
        
        ## Add collision with rectangle walls 
        ## !! only support rectangles walls for now.
        collision_wWall = torch.zeros_like(collision_c).bool() # (batch, K)
        for wall in wall_polygons:
            xmin, xmax, ymin, ymax = wall[:,0].min(), wall[:,0].max(), wall[:,1].min(), wall[:,1].max()
            xmin, xmax, ymin, ymax = xmin-r_radius, xmax+r_radius, ymin-r_radius, ymax+r_radius
            collision_flag = (next_r_state[...,0]>=xmin)*(next_r_state[...,0]<=xmax) * (next_r_state[...,1]>=ymin)*(next_r_state[...,1]<=ymax)
            collision_wWall = torch.logical_or(collision_flag, collision_wWall) # if there was collision with any wall.
        # Consider collision only once if collision with both wall and human.
        collision_c = torch.where(collision_wWall, collision_penalty, collision_c)
        
        ########## Discomfort costs based on ego's state vector ###########
        # if collision_c == 0 and dmin<discomfort_dist: # don't count if collision
        #     discomfort_c = (dmin - discomfort_dist) * discomfort_penalty_factor * self.time_step
        
        discomfort_c = torch.where(torch.logical_and(collision_c == 0, closest_dist < discomfort_dist),
                                   (discomfort_dist-closest_dist) * discomfort_penalty_factor * self.time_step,
                                   torch.tensor(0.0))
        
        ########### Costs based on ego's state vector ###########
        ## reaching goal
        goal_reaching_c = torch.tensor(0.0).repeat(r_state.shape[1])
        goal_reaching_c = torch.where(torch.logical_and(torch.linalg.norm(next_r_state - r_goal, axis=-1) < r_radius, collision_c==0.),
                                      torch.tensor(self.config.reward.success_reward),
                                      torch.tensor(0.0)).view(-1)
        
        ## moving towards the goal
        potential_cur = torch.linalg.norm(r_state - r_goal, axis=-1)
        potential_next = torch.linalg.norm(next_r_state - r_goal, axis=-1)
        potential_c = torch.tensor(2.) * (potential_next-potential_cur) # (batch, K) # penalty if moving away from the goal (potential increases)
        # No potential cost if the robot is already at the goal
        potential_c = torch.where(goal_reaching_c > 0., torch.tensor(0.0), potential_c).view(-1)
        
        # # # TODO: decoded should only be estimating the unobserved areas. Otherwise, it should be the ground truth or zero.        
        # # # For now just depress it manually.
        if self.config.reward.disambig_reward_flag:
            # ## Compute disambiguating cost, which gives a "reward" for reducing the entropy in the following time step
            # ## Aims to reduce the uncertainty in the unobserved areas & confirm the PaS inference.
            # # Compute PaS inference grid, estimation only in the unobserved areas and with the ground truth in the observed areas, in the time current timestep (t=0)
            
            if t == 0:
                map_xy = self.curr_map_xy.repeat(next_r_state.shape[1],1,1,1)
                map_xy = [map_xy[:,0], map_xy[:,1]]
                sensor_grid = self.curr_sensor_grid.repeat(next_r_state.shape[1],1,1)
                self.H_cur, self.disambig_R_map, self.disambig_W_map = compute_disambig_cost(r_state.squeeze(0), self.curr_map_xy, sensor_grid, sensor_grid, decoded_in_unknown, self.r_goal, self.config)
                
            # next_empty_grid = torch.zeros_like(self.curr_sensor_grid)
            ## TODO: make the untransferred areas to be unknown (0.5) in the transferred grid?
            # curr_map_xy = [self.curr_map_xy[0].unsqueeze(0), self.curr_map_xy[1].unsqueeze(0)]
            
            # # Transfer the next time-step sensor_grid information to the current map_xy since the current map is the region of interest to determine the next action. 
            # # Might not need to transfer the grid though, but just represent the next states in the current map.
            # transferred_next_sensor_grid = Transfer_to_EgoGrid(next_map_xy, next_sensor_grid, curr_map_xy, next_empty_grid) # decoded in next time step (t=1)      

            next_human_pos, human_id, next_robot_pos = dict_update(next_h_state, next_r_state, self.config, self.vis_ids) 
            next_label_grid, _, _ = generateLabelGrid_mppi(self.curr_map_xy, next_human_pos, h_radius.squeeze(0), human_id, next_robot_pos, wall_polygons, res=grid_res)
            # next_map_xy = [next_x_local.to(next_label_grid.dtype), next_y_local.to(next_label_grid.dtype)]
            next_sensor_grid = generateSensorGrid_mppi(next_label_grid, next_human_pos, h_radius.squeeze(0), next_robot_pos, self.curr_map_xy, FOV_radius, wall_polygons, res=grid_res)
            H_next, disambig_R_map_next, disambig_W_map_next = compute_disambig_cost(next_r_state.squeeze(0), self.curr_map_xy, next_sensor_grid.clone(), self.curr_sensor_grid, decoded_in_unknown, self.r_goal, self.config)
            
            # If next entropy is reduced big (H_cur-H_next), then lower the cost. If not, increase the cost.
            ## Clip the disambig_c to be positive
            disambig_c = (H_next-self.H_cur) * self.config.reward.disambig_factor  # -(-H_next+H_cur) # torch.clamp(H_next-H_cur,min=0.0) 
            
            # make the dismbig_c 0 if the discomfort_c or collision is high
            disambig_c = torch.where(torch.logical_or(discomfort_c>0, collision_c>0), torch.tensor(0.0), disambig_c)    
            
            # make the disambig_c flag based regardless of the number of grid cells
            # disambig_c = torch.where(disambig_c<0, -self.config.reward.disambig_factor, torch.tensor(0.0)).squeeze(0)       
            disambig_c = torch.where(disambig_c<0, -self.config.reward.disambig_factor, torch.tensor(0.0)).squeeze(0)    
            
            # # Sanity check
            # if len(collision_sample_idx) > 0: # visualize collision samples
                # print(len(collision_sample_idx))
                # k = collision_sample_idx[0]
            # if len(discomfort_c.nonzero()) > 0: # visualize discomfort samples
            #     # discomfort_sample_idx = torch.where(discomfort_c>0)
            #     # k = discomfort_sample_idx[0][0]
                
            #     # TODO: Compute the current time step's label and sensor grid for visualization
                # print(next_label_grid[k][0].shape, next_sensor_grid[k].shape, transferred_next_sensor_grid[k].shape, disambig_R_map_next[k].shape, decoded[0,0].shape, self.curr_label_grid.shape, self.curr_sensor_grid.shape)
            
            # for k in torch.where(next_sensor_grid==1.)[0][:2]:
            # # for k in range(1):
            #     curr_map_x, curr_map_y = self.curr_map_xy
            #     vis_grids = [[curr_map_x, curr_map_y, next_label_grid[k][0], "Next label grid"]]
            #     # vis_grids = [[next_x_local[k], next_y_local[k], next_label_grid[k][1], "Next id grid"]]
            #     vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[k], "Next sensor grid"])
            # #     # vis_grids.append([next_x_local[k], next_y_local[k], collision_cost_grid[k], "Collision cost grid"])
            # #     # vis_grids.append([next_x_local[k], next_y_local[k], discomfort_cost_grid[k], "Discomfort cost grid"])
            #     vis_grids.append([curr_map_x, curr_map_y, disambig_W_map_next[k], "Next disambig W map"])
            #     vis_grids.append([curr_map_x, curr_map_y, disambig_R_map_next[k], "Next disambig R map"])
            #     vis_grids.append([curr_map_x, curr_map_y, decoded[0,0], " Decoded grid"])
            #     vis_grids.append([curr_map_x, curr_map_y, decoded_in_unknown, "Decoded grid in unknown area"])
            #     # vis_grids.append([next_x_local[k], next_y_local[k], transferred_decoded_in_unknown[k], "Transferred decoded grid to future coordinate"])
            #     # vis_grids.append([next_x_local[k], next_y_local[k], integrated_next_sensor[k], "PaS Integrated next sensor grid "])
            # #     # vis_grids.append([curr_map_x, curr_map_y, self.curr_label_grid, "Current label grid"])
            #     vis_grids.append([curr_map_x, curr_map_y, self.curr_sensor_grid, "Current sensor grid"])
                
            #     # vis_grids.append([curr_map_x, curr_map_y, decoded[0,k], "PaS inference grid at t=0"])
            #     # vis_grids.append([next_x_local[k], next_y_local[k], next_pas_inf_grid[k], "PaS inference grid at t=k"])
            #     # To debug the Transfer_to_EgoGrid
            #     # empty_grid2 = torch.zeros_like(sensor_grid)
            #     # pas_inf_grid = Transfer_to_EgoGrid(torch.stack(self.curr_map_xy).permute(1,0,2,3)[0], self.curr_sensor_grid[0], map_xy, empty_grid2, grid_res) # decoded in current time (t=0) 
            #     # vis_grids.append([next_x_local[k], next_y_local[k], pas_inf_grid[k], "Current sensor grid at t=k"])
            #     self.visualize(vis_grids, self.r_goal, disambig_c[k])
            # # self.visualize_traj(r_state, next_r_state, self.r_goal)
        else:
            self.disambig_R_map = self.curr_sensor_grid.repeat(next_r_state.shape[1],1,1)
            if 'pas' in self.config.robot.policy:
                self.disambig_R_map = torch.where(self.disambig_R_map==0.5, decoded[0,0], torch.tensor(0.0))
            disambig_c = torch.tensor(0.0).repeat(r_state.shape[1]).to(self.device)
            # self.visualize_traj(r_state, next_r_state, self.r_goal)

        
        ## Check if there were any collisions                
        # goal_reaching_c = 0 or -5
        # potential_c: -0.99~ 0.97
        # collision_c: [0, 10, 30, 70,110.,200.]  -> make this regardless of the number of occupancy cells. Now 0 or 10.
        # discomfort_c: [0., 10, 30, 70.] -> make this regardless of the number of occupancy cells. No 0 or 2.
        # disambig_c: 0 or [-8.3, ]
        
        # TODO: check collision with PaS estimation
        any_seen_decoded = self.curr_sensor_grid[decoded[0,0]>0.4]==1 # (batch, K, 1)
        # print(torch.any(decoded[0,0]>0.4))
        # print(self.curr_sensor_grid[decoded[0,0]>0.4])
        # print(torch.any(any_seen_decoded))

        # Once part of potential occluded_agent is seen in the current timestep, no need to consider PaS_colilsion that is very conservative.
        if 'pas' in self.config.robot.policy and not torch.any(any_seen_decoded):
            
            # flatten decoded_in_unknown and its x,y coordinate and obtain the estimated occupancy cells
            decoded_in_unknown = decoded_in_unknown.view(-1)
            decoded_x = self.curr_map_xy[0].view(-1)
            decoded_y = self.curr_map_xy[1].view(-1)
            
            PaS_occupied_indices = torch.where(decoded_in_unknown>0.4)
            if len(PaS_occupied_indices[0])>0:
                
                ### PaS collision cost
                PaS_occupied_x = decoded_x[PaS_occupied_indices]
                PaS_occupied_y = decoded_y[PaS_occupied_indices]
                PaS_occupied_xy = torch.stack([PaS_occupied_x, PaS_occupied_y], dim=-1) # (N_indices, 2)
                
                PaS_occupied_prob = decoded_in_unknown[PaS_occupied_indices] # (N_indices)
                
                # compute collision cost for PaS occupied cells.
                dist_to_PaS_occupied = torch.linalg.norm(PaS_occupied_xy.view(1,1,-1,2) - next_r_state.unsqueeze(2), dim=-1) #(1,1,N_indices, 2) - (batch,K,1, 2) -> (batch,K,N_indices,2) -> (batch,K,N_indices)
                dmin_to_PaS_occupied =  torch.min(dist_to_PaS_occupied, axis=-1) # (batch,K)
                dmin_PaS_indicies = dmin_to_PaS_occupied.indices # (batch,K)
                
                closest_dist_PaS = dmin_to_PaS_occupied.values - r_radius # (batch,K)
                
                ### PaS collision penalty            
                # PaS_collision_c = torch.where(closest_dist_PaS<discomfort_dist*5, # For PaS, make "potential collision" distance is larger due to uncertainty.
                #                             torch.tensor(collision_penalty)*PaS_occupied_prob[dmin_PaS_indicies],
                #                             torch.tensor(0.0))     
                
                ## PaS high speed penalty
                ### Slow down the robot when there's potential occluded obstacle. This keeps the robot goes straight, instead of taking detour. like PaS_collision_cost
                PaS_collision_c = torch.where(closest_dist_PaS<discomfort_dist*3, # For PaS, make "potential collision" distance is larger due to uncertainty.
                                            (next_state[...,3])**2*PaS_occupied_prob.max()*0.3,
                                            torch.tensor(0.0))   
                # PaS_collision_c = torch.tensor(0.0).repeat(r_state.shape[1]).to(self.device)
            else:
                PaS_collision_c = torch.tensor(0.0).repeat(r_state.shape[1]).to(self.device)
        else:
            PaS_collision_c = torch.tensor(0.0).repeat(r_state.shape[1]).to(self.device)

        ### Lane keeping cost
        # penalize if the robot deviates from the lane (x-=0)
        lane_keeping_c = next_r_state[...,0]**2 *5       
        
        ## Check if there were any collisions                
        # goal_reaching_c = 0 or -5
        # potential_c: -0.99~ 0.97
        # collision_c: [0, 10, 30, 70,110.,200.]  -> make this regardless of the number of occupancy cells. Now 0 or 10.
        # discomfort_c: [0., 10, 30, 70.] -> make this regardless of the number of occupancy cells. No 0 or 2.
        # disambig_c: 0 or [-8.3, ]
        
        goal_reaching_c = goal_reaching_c.view(-1)
        potential_c = potential_c.view(-1)
        collision_c = collision_c.view(-1)
        discomfort_c = discomfort_c.view(-1)
        disambig_c = disambig_c.view(-1)
        PaS_collision_c = PaS_collision_c.view(-1)
        lane_keeping_c = lane_keeping_c.view(-1)
        # PaS_discomfort_c = PaS_discomfort_c.view(-1)        
        
        cost = goal_reaching_c.float() + potential_c + collision_c.float() + discomfort_c.float() + disambig_c.float() # + PaS_collision_c.float() + lane_keeping_c.float() #+ PaS_discomfort_c.float()
        
        least_cost_idx = torch.argmin(cost)
        if torch.any(potential_c <0):
        # if torch.any(disambig_c<0):
            print(potential_c)
            # print(goal_reaching_c.float(), collision_c.float(), discomfort_c.float())
            # print(lane_keeping_c, PaS_collision_c, potential_c, disambig_c)
            # print('least cost:', lane_keeping_c[least_cost_idx], PaS_collision_c[least_cost_idx], potential_c[least_cost_idx], disambig_c[least_cost_idx])
            # print(goal_reaching_c[least_cost_idx], collision_c[least_cost_idx], discomfort_c[least_cost_idx], disambig_c[least_cost_idx])
            # pdb.set_trace()        
        # ## Visualizing critical disambiguation moments with the disambiguating/nondisambiguating actions labeled.
        # # interest_indices = torch.where(PaS_discomfort_c!=0)[0] #torch.where(torch.logical_and(torch.abs(H_next-self.H_cur) > 0,collision_c==0.))[1]
        # # TODO: update both the interest_indices and lowest_cost_ind
        # interest_indices = torch.where(disambig_c!=0)[0]
        # lowest_interest_indices = torch.where(disambig_c==disambig_c.min())[0]
        # lowest_cost_ind = torch.argmin(cost)
        # # print('disambig_c',disambig_c)
        # # print('lowest cost ind:', lowest_cost_ind)
        # # if torch.any(torch.abs(H_next-self.H_cur)!=0):
        # #     pdb.set_trace()
        # # print(disambig_c, H_next, self.H_cur)
        # if len(interest_indices) > 0:
            
        #     # disambig_indices = disambig_indices[1]
        #     ## Visualize when the entropy changes.
        #     # disambig_indices = torch.where(torch.logical_and(torch.abs(H_next-self.H_cur) > 0,collision_c==0.))[0]
        #     # for k in disambig_indices:
        #     k = 0
        #     # curr_map_x, curr_map_y = self.curr_map_xy
        #     # vis_grids = [[curr_map_x, curr_map_y, next_label_grid[k][0], "Next label grid"]]
        #     # vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[k], "Next sensor grid1"])
        #     # vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[1], "Next sensor grid2"])
        #     # vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[2], "Next sensor grid3"])
        #     # vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[3], "Next sensor grid4"])
        #     # vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[4], "Next sensor grid5"])
        #     # vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[5], "Next sensor grid6"])
        #     # vis_grids.append([curr_map_x, curr_map_y, disambig_W_map_next[k], "Next disambig W map"])
        #     # vis_grids.append([curr_map_x, curr_map_y, disambig_R_map_next[k], "Next disambig R map"])
        #     # vis_grids.append([curr_map_x, curr_map_y, decoded[0,0], "Decoded grid"])
        #     # vis_grids.append([curr_map_x, curr_map_y, decoded_in_unknown, "Decoded grid in unknown area"])
        #     # vis_grids.append([curr_map_x, curr_map_y, self.curr_sensor_grid, "Current sensor grid"])
        #     # # visualize_robot_trajectory as well
        #     # self.visualize(vis_grids, self.r_goal, disambig_c[k])
            
        #     cur_h_states = ob[0, [0], t, :, :2] # (1, human_num, 2)
        #     self.visualize_disambig_traj(r_state, next_r_state, self.r_goal, interest_indices, self.curr_map_xy, self.curr_sensor_grid, wall_polygons, cur_h_states, lowest_interest_indices, lowest_cost_ind, cost)
        #     # pdb.set_trace()
        
        
        # discomfort_indices = torch.where(torch.logical_and(discomfort_c> 0,collision_c==0.))[1]
        # print(discomfort_c, dmin.values+ hr+r_radius, discomfort_dist)
        # if len(discomfort_indices) > 0:
        #     pdb.set_trace()
        #     # disambig_indices = disambig_indices[1]
        #     ## Visualize when the entropy changes.
        #     # disambig_indices = torch.where(torch.logical_and(torch.abs(H_next-self.H_cur) > 0,collision_c==0.))[0]
        #     for k in discomfort_indices:
        #         curr_map_x, curr_map_y = self.curr_map_xy
        #         vis_grids = [[curr_map_x, curr_map_y, next_label_grid[k][0], "Next label grid"]]
        #         vis_grids.append([curr_map_x, curr_map_y, next_sensor_grid[k], "Next sensor grid"])
        #         vis_grids.append([curr_map_x, curr_map_y, disambig_W_map_next[k], "Next disambig W map"])
        #         vis_grids.append([curr_map_x, curr_map_y, disambig_R_map_next[k], "Next disambig R map"])
        #         vis_grids.append([curr_map_x, curr_map_y, decoded[0,0], "Decoded grid"])
        #         vis_grids.append([curr_map_x, curr_map_y, decoded_in_unknown, "Decoded grid in unknown area"])
        #         vis_grids.append([curr_map_x, curr_map_y, self.curr_sensor_grid, "Current sensor grid"])
        #         # visualize_robot_trajectory as well
        #         self.visualize(vis_grids, self.r_goal, disambig_c[k])
            
        #     self.visualize_disambig_traj(r_state, next_r_state, self.r_goal, discomfort_indices, self.curr_map_xy, self.curr_sensor_grid)
        #     pdb.set_trace()
        
        # pdb.set_trace()
        # print('t:', t, 'cost', cost[torch.where(disambig_c>0)], 'goal', goal_reaching_c[torch.where(disambig_c>0)], 'potential', potential_c[torch.where(disambig_c>0)], 'collision', collision_c[torch.where(disambig_c>0)], 'discomfort', discomfort_c[torch.where(disambig_c>0)], 'disambig', disambig_c[torch.where(disambig_c>0)])
        
        # if torch.any(disambig_c != 0) or torch.any(discomfort_c > 0) or torch.any(collision_c>0):
        #     mask = torch.where(torch.logical_or(discomfort_c>0, collision_c>0))
        #     print('goal',goal_reaching_c[mask], 'potent',potential_c[mask], 'colli',collision_c[mask], 'discomfort',discomfort_c[mask], 'disambig', disambig_c[mask])
        # self.next_map_xy = next_map_xy
        # self.next_sensor_grid = next_sensor_grid
        # if self.step == 40:
        #     self.costs.append(cost)
            # if len(self.costs)>self.lookahead_steps-1:
            #     pdb.set_trace()            
        return cost
    
    def plan(self, obs, decoded, wall_polygons, config):
        """
        obs: full sequence (batch, sequence+lookahead_steps, 1+human_num, 7)
        decoded
        !! Not supporting batch yet.
        
        return: trajectory (1+human_num, (lookahead_steps)*4)
        """
        # self.costs = []
        # self.step += 1
        # del self.disambig_R_map
        del self.curr_label_grid, self.curr_sensor_grid, self.curr_map_xy, self.vis_ids, self.disambig_R_map
        sequence = config.pas.sequence
        state = obs['mppi_vector'][0,sequence-1, 0].clone() # (1, 7) robot's current state
        ob = obs['mppi_vector'][0,sequence-1:, 1:].clone() # Using vector states (1+lookahead_steps, human_num, 7) human's current and future states
        self.curr_label_grid = obs['label_grid'][0].clone() # (1, 2,  H, W)
        self.curr_sensor_grid = obs['grid'][0, -1].clone()  # (1,sequence, H, W)
        self.curr_map_xy = obs['grid_xy'][0].clone() 
        vis_ids = torch.unique(obs['vis_ids'][0]).clone() 
        wall_polygons = [torch.tensor(wall_polygons[i]) for i in range(len(wall_polygons))] 
        if len(vis_ids) > 1:
            self.vis_ids = vis_ids[1:] # remove dummy id (-1)
        else:
            self.vis_ids = torch.tensor([]).to(self.device)
            
        self.r_goal = state[4:6]# (batch, 2)
        propagated_states, action = self.mppi_gym.command(state, ob, decoded, wall_polygons)
        
        ## Copy obs['mppi_vector'] and assign the propated_states to the robot's states
        final_traj = obs['mppi_vector'][:, sequence-1:, :, :4].clone() # (1, 1+lookahead_steps, 1+human_num, 4)
        
        
        ### acc.py ###
        final_traj[0,1:,0, :2] = propagated_states[...,:2] # (lookahead_steps, 2) 
        # convert acceleration action[...,0], to velocity
        prev_states = torch.vstack([state[...,:4].unsqueeze(0), propagated_states[:-1]]) # (1+lookahead_steps, 4)
        v = prev_states[...,3] + action[...,0] * self.time_step
        action_vr = action.clone()
        action_vr[...,0] = v  
        final_traj[0,1:,0, 2:] = action_vr # velocity # (lookahead_steps, 2)
        #### 
                
        final_traj = final_traj.permute(0,2,1,3).reshape(-1,1) # (1, 1+human_num, 1+lookahead_steps, 4) 
        return final_traj, self.disambig_R_map
        
        
        
            
        
        
        
        
        
        
        
    
