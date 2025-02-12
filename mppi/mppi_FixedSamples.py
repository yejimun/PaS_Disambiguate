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

# TODO: visualize the next states from sampled actions and debug the collision cases.
# TODO: debug why the robot's future trajectories are so close. 
# TODO: improve adaptive sampling to be more effective.
# TODO: improve transfer_to_ego_grid to be more efficient. Maybe try using https://github.com/getkeops/keops/tree/main.


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
            self.U = self.noise_dist.sample((self.K, self.T)).to(self.d) #torch.zeros((self.T,self.nu)).to(self.d)#self.noise_dist.sample((self.T,))
            # self.noise = self.noise_dist.sample((self.K, self.T)).to(self.d)
            # self.noise = self.noise_dist.sample((self.K,1)).repeat(1, self.T, 1)

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
    def _running_cost(self, state, ob, next_state, decoded, t):
        return self.running_cost(state, ob, next_state, decoded, t)# if self.step_dependency else self.running_cost(state, u)

    def shift_nominal_trajectory(self):
        """
        Shift the nominal trajectory forward one step
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

    def command(self, state, ob, decoded, shift_nominal_trajectory=False, info=None):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :param shift_nominal_trajectory: Whether to roll the nominal trajectory forward one step. This should be True
        if the command is to be executed. If the nominal trajectory is to be refined then it should be False.
        :returns action: (nu) best action
        """
        self.info = info
        if shift_nominal_trajectory:
            self.shift_nominal_trajectory()
        return self._command(state, ob, decoded)
    
    def _compute_weighting(self, cost_total):
        beta = torch.min(cost_total)
        self.cost_total_non_zero = _ensure_non_zero(cost_total, beta, 1 / self.lambda_)
        eta = torch.sum(self.cost_total_non_zero)
        self.omega = (1. / eta) * self.cost_total_non_zero
        return self.omega
    def _command(self, state, ob, decoded):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        self.ob = ob.to(dtype=self.dtype, device=self.d)
        self.decoded = decoded.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()
        
        # self._compute_weighting(cost_total)
        # perturbations = torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)

        # self.U = self.U  + perturbations
        # action = self.U[:self.u_per_command]
        # # reduce dimensionality if we only need the first command
        # if self.u_per_command == 1:
        #     action = action[0]
        
        action = self.perturbed_action[torch.argmin(cost_total)]
            
        propagated_states = []
        ## Rollouts with the final action
        for a in range(self.u_per_command):
            if self.kinematics == "holonomic":
                state = self._dynamics(state[:2], action[a])
                propagated_states.append(state)
            elif self.kinematics == "unicycle":
                state = self._dynamics(state[:3], action[a])
                propagated_states.append(state[:2])
            
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
            state = state.repeat(self.M, 1, 1)[:, :, :3] 
        
        
        # (batch, K, pred_horizon, human_num, 7) # repeated obs with pred_horizon
        ob = self.ob.unsqueeze(0).unsqueeze(0).repeat(self.M, K, 1, 1, 1) # vector obs
                
        states = []
        actions = []
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t]#.repeat(self.M, 1, 1)
            next_state = self._dynamics(state, u) # (1,50,2)

            c = self._running_cost(state, ob, next_state, self.decoded, t) 
            state = next_state
            cost_samples = cost_samples + c / (t+1)
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
        perturbed_action = self.U #+ self.noise
        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        # self._compute_perturbed_action_and_noise()
        # if self.noise_abs_cost:
        #     action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
        #     # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
        #     # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
        #     # nomial trajectory.
        # else:
        #     action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv  # Like original paper

        rollout_cost, self.states, actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions = actions / self.u_scale

        # action perturbation cost
        perturbation_cost = 0. #torch.sum(self.U * action_cost, dim=(1, 2))
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
                    cu = u.clone()
                    cu[:,0] = torch.max(torch.min(u[:,0], self.u_max[0]), self.u_min[0])
                    cu[:,1] = torch.max(torch.min(u[:,1], self.u_max[1]), self.u_min[1])
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


# def run_mppi(mppi, env, retrain_dynamics, retrain_after_iter=50, iter=1000, render=True):
#     dataset = torch.zeros((retrain_after_iter, mppi.nx + mppi.nu), dtype=mppi.U.dtype, device=mppi.d)
#     total_reward = 0
#     for i in range(iter):
#         state = env.unwrapped.state.copy()
#         command_start = time.perf_counter()
#         action = mppi.command(state, shift_nominal_trajectory=False)
#         elapsed = time.perf_counter() - command_start
#         res = env.step(action.cpu().numpy())
#         s, r = res[0], res[1]
#         total_reward += r
#         logger.debug("action taken: %.4f cost received: %.4f time taken: %.5fs", action, -r, elapsed)
#         if render:
#             env.render()

#         di = i % retrain_after_iter
#         if di == 0 and i > 0:
#             retrain_dynamics(dataset)
#             # don't have to clear dataset since it'll be overridden, but useful for debugging
#             dataset.zero_()
#         dataset[di, :mppi.nx] = torch.tensor(state, dtype=mppi.U.dtype)
#         dataset[di, mppi.nx:] = action
#     return total_reward, dataset


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
        self._init_mppi()
        
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
        else:
            raise ValueError("Invalid kinematics")
        
        
        self.mppi_gym = MPPI(self.dynamics, self.kinematics, self.running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
                         lambda_=lambda_, device=self.device, u_min=torch.tensor(ACTION_LOW, dtype=torch.double, device=self.device),
                         u_max=torch.tensor(ACTION_HIGH, dtype=torch.double, device=self.device), u_per_command=self.lookahead_steps,
                        step_dependent_dynamics=False, sample_null_action=True)
        
    def dynamics(self, state, action):
        """
        state: robot's current state (K,2)
        """
        # print(state.shape, action.shape)
        
        # State only containes robot's position
        if self.kinematics == "holonomic":
            next_r_state = state + action * self.time_step
        elif self.kinematics == "unicycle":
            next_theta = (state[...,2] + action[...,1]) % (2 * torch.pi)
            next_vx = action[...,0] * torch.cos(next_theta) 
            next_vy = action[...,0] * torch.sin(next_theta) 
            next_x = state[...,0] + next_vx * self.time_step
            next_y = state[...,1] + next_vy * self.time_step
            next_r_state = torch.stack([next_x, next_y, next_theta], dim=-1)
        else:
            raise ValueError("Invalid kinematics")        
        return next_r_state
    
    def visualize_traj(self,r_state, next_r_state, goal_pos):
        """
        r_state: robot's current state (K,2)
        next_r_state: robot's next state (K,2)
        """
        ## Sanity check: Visualize the human and robot's positions in circle and masks in multiple plots
        # make visualizaton directory
        if not os.path.exists("mppi_sanitycheck"):
            os.makedirs("mppi_sanitycheck")
        # count how many is already saved
        k = len(os.listdir("mppi_sanitycheck")) 
        
        fig, ax = plt.subplots(1, figsize=(10, 10))
        
        ax.tick_params(labelsize=16)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        ax.plot(r_state[0,:,0].cpu(), r_state[0,:,1].cpu(), 'bo', markersize=10)
        ax.plot(next_r_state[0,:,0].cpu(), next_r_state[0,:,1].cpu(), 'ro', markersize=10)
        ax.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'go', markersize=10)
        ax.set_title("Robot's current and sampled next state")
        ax.legend(["Current state", "Next state", "Goal"])
        
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
        plt.savefig("mppi_sanitycheck/vis_traj_" + str(k)+".png")
        

    
    def visualize(self,vis_grids, goal_pos, disambig_c, r_state, next_r_state, next_h_state):
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
        # visualize the robot and human's positions
        ax0 = axes.flatten()[0]
        ax0.tick_params(labelsize=16)
        ax0.set_xlim(-6, 6)
        ax0.set_ylim(-6, 6)
        ax0.set_xlabel('x(m)', fontsize=16)
        ax0.set_ylabel('y(m)', fontsize=16)

        ax0.plot(r_state[0].cpu(), r_state[1].cpu(), 'go', markersize=10)
        ax0.plot(next_r_state[0].cpu(), next_r_state[1].cpu(), 'go', markersize=10)
        for i in range(next_h_state.shape[0]):
            color = ['b', 'c', 'm', 'y', 'k']*5
            ax0.plot(next_h_state[i,0].cpu(), next_h_state[i,1].cpu(), color[i], markersize=10)
        ax0.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'ro', markersize=10)
        ax0.set_title("Robot's current and sampled next state")
        
        for ax, grid_info in zip(axes.flatten()[1:], vis_grids):
            # use contourf to create a filled contour plot
            map_x, map_y, grid, title = grid_info
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            if "id grid" in title:
                print("id in ", title)
                CS = ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='tab20', levels=np.linspace(0, 6, 6))
                # add the color bar
                cbar = plt.colorbar(CS)
            else:
                ax.contourf(map_x.cpu().numpy(),map_y.cpu().numpy(), grid.cpu().numpy(), cmap='binary', levels=np.linspace(0, 1, 9))
            ax.plot(goal_pos[0].cpu(), goal_pos[1].cpu(), 'ro', markersize=10)
            ax.set_title(title)

        # save the figure
        # plt.title("Visible ids: " + str(self.vis_ids))
        plt.savefig("mppi_sanitycheck/vis_" + str(k)+".png")
        

    def running_cost(self, state, ob, next_state, decoded, t):
        """
        state: robot's state at t (batch, K, 2). Note that this is not the current state but a rolled state.
        ob: human's state at curr_t:curr_t+look_ahead (batch, K, 1+lookahead_steps, human_num, 7) 
        """
        if self.kinematics == "holonomic":
            r_state = state
            next_r_state = next_state # self.dynamics(state, action)#.unsqueeze(-2)
        elif self.kinematics == "unicycle":
            r_state = state[:,:,:2]
            next_r_state = next_state[:,:,:2]
            
        r_radius = torch.tensor(self.config.robot.radius, device=self.device).repeat(r_state.shape[0],r_state.shape[1]) # (batch, K)
        r_goal = torch.tensor(self.r_goal, device=self.device).repeat(r_state.shape[0],r_state.shape[1],1) # (batch, K)
        
        ########### Costs based on ego's observation OGM ###########
        # Collision cost (in the next state) 
        collision_penalty = self.config.reward.collision_penalty
        next_h_state = ob[:, :, t+1, :, :2] # (batch, K, human_num, 2)
        h_radius = ob[0, :, t+1, :, -1]
        
        # Generate label grid
        FOV_radius = self.config.robot.FOV_radius
        grid_res = self.config.pas.grid_res
        
        # TODO: update to consider batch, not just number of samples (K)
        ## Get sensor grid at time lookahead t+1
        next_human_pos, human_id, next_robot_pos = dict_update(next_h_state, next_r_state, self.config, self.vis_ids) 
        next_label_grid, next_x_local, next_y_local = generateLabelGrid(next_human_pos, h_radius, human_id, next_robot_pos)
        next_map_xy = [next_x_local.to(next_label_grid.dtype), next_y_local.to(next_label_grid.dtype)]
        next_sensor_grid = generateSensorGrid(next_label_grid, next_human_pos, h_radius, next_robot_pos, next_map_xy, FOV_radius, res=grid_res)
         
        # Create collision mask for each (batch, K): Ego in the center
        robot_mask = torch.sqrt((next_x_local - next_r_state[:,:,0].reshape(-1,1,1))**2 + (next_y_local - next_r_state[:,:,1].reshape(-1,1,1))**2) < r_radius.reshape(-1, 1, 1) # (K, height, width)

        # Compute collision cost checking collision using "sensor_grid" (observation, not the ground truth) and collision mask.
        next_occupancy_grid = next_sensor_grid.clone()
        next_occupancy_grid = torch.where(next_occupancy_grid==1., torch.tensor(1.0), torch.tensor(0.0))
        collision_cost_grid = next_sensor_grid * robot_mask
        # penalty based on the number of collided grid cells
        # collision_c = torch.sum(collision_cost_grid, dim=(1,2)) * collision_penalty 
        # penalty based on collision flag, not the number of grid cells
        collision_c = torch.any(collision_cost_grid>0, dim=(1,2)) * collision_penalty
        
        # Discomfort costs
        discomfort_dist = self.config.reward.discomfort_dist
        discomfort_penalty_factor = self.config.reward.discomfort_penalty_factor
        
        discomfort_mask = torch.sqrt((next_x_local - next_r_state[:,:,0].reshape(-1,1,1))**2 + (next_y_local - next_r_state[:,:,1].reshape(-1,1,1))**2) < r_radius.view(-1, 1, 1)+discomfort_dist
        # # Don't count discomfort if there is a collision
    
        discomfort_cost_grid = next_occupancy_grid * discomfort_mask
        # penalty based on the number of discomfort grid cells
        discomfort_c = torch.sum(discomfort_cost_grid, dim=(1,2))# * discomfort_penalty_factor
        # penalty based on discomfort flag, not the number of grid cells
        discomfort_c = torch.where(torch.logical_and(discomfort_c>0., collision_c==0.), discomfort_penalty_factor, torch.tensor(0.))     
        
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
        
        if self.config.reward.disambig_reward_flag:
            # ## Compute disambiguating cost, which gives a "reward" for reducing the entropy in the following time step
            # ## Aims to reduce the uncertainty in the unobserved areas & confirm the PaS inference.
            # # Compute PaS inference grid, estimation only in the unobserved areas and with the ground truth in the observed areas, in the time current timestep (t=0)
            
            # # # TODO: decoded should only be estimating the unobserved areas. Otherwise, it should be the ground truth or zero.        
            # # # For now just depress it manually.
            decoded_in_unknown = torch.where(self.curr_sensor_grid==0.5, decoded[0,0], torch.tensor(0.0))
            
            if t == 0:
                map_xy = self.curr_map_xy.repeat(next_r_state.shape[1],1,1,1)
                map_xy = [map_xy[:,0], map_xy[:,1]]
                sensor_grid = self.curr_sensor_grid.repeat(next_r_state.shape[1],1,1)
                self.H_cur, self.disambig_R_map, self.disambig_W_map = compute_disambig_cost(r_state.squeeze(0), self.curr_map_xy, sensor_grid, sensor_grid, decoded_in_unknown, self.r_goal, self.config)
                
            # else:
            #     map_xy = self.next_map_xy
            #     sensor_grid = self.next_sensor_grid
            #     # ## Get sensor grid at time lookahead t
            #     # h_state_c = ob[:, :, t, :, :2]
            #     # # TODO: update to consider batch, not just number of samples (K)
            #     # human_pos, human_id, robot_pos = dict_update(h_state_c, r_state, self.config) 
            #     # label_grid, x_local, y_local = generateLabelGrid(human_pos, h_radius, human_id, robot_pos)
            #     # map_xy = [x_local.to(label_grid.dtype), y_local.to(label_grid.dtype)]            
            #     # sensor_grid = generateSensorGrid(label_grid, human_pos, h_radius, robot_pos, map_xy, FOV_radius, res=grid_res)
            #     # # Obtain the pas inference grid: extract only the unknown areas (0.5) of the sensor grid from the decoded

            # ### TODO: transfer pas_integrated_grid and next_pas_integrated_grid to the same coordinate (curr_map_xy??)
            # empty_grid = torch.zeros_like(self.curr_sensor_grid)
            # transferred_sensor_grid = Transfer_to_EgoGrid(map_xy, sensor_grid, self.curr_map_xy, empty_grid, grid_res) # decoded in current time (t=0) 
            # # # Mask 
            # # pas_mask = torch.logical_and(decoded_in_unknown==1. and transferred_sensor_grid==0.5)
            # # pas_integrated_grid = torch.where(pas_mask, decoded_in_unknown, transferred_sensor_grid)
            
            next_empty_grid = torch.zeros_like(self.curr_sensor_grid)
            ## TODO: make the untransferred areas to be unknown (0.5) in the transferred grid?
            transferred_next_sensor_grid = Transfer_to_EgoGrid(next_map_xy, next_sensor_grid, self.curr_map_xy, next_empty_grid, grid_res) # decoded in next time (t=1)      

            H_next, disambig_R_map_next, disambig_W_map_next = compute_disambig_cost(next_r_state.squeeze(0), self.curr_map_xy, transferred_next_sensor_grid, self.curr_sensor_grid, decoded_in_unknown, self.r_goal, self.config)
            
            # If next entropy is reduced big (H_cur-H_next), then lower the cost. If not, increase the cost.
            ## Clip the disambig_c to be positive
            disambig_c = (H_next-self.H_cur) * self.config.reward.disambig_factor  # -(-H_next+H_cur) # torch.clamp(H_next-H_cur,min=0.0) 
            
            # make the dismbig_c 0 if the discomfort_c or collision is high
            disambig_c = torch.where(torch.logical_or(discomfort_c>0, collision_c>0), torch.tensor(0.0), disambig_c)    
            
            # make the disambig_c flag based regardless of the number of grid cells
            disambig_c = torch.where(disambig_c>0, self.config.reward.disambig_factor, torch.tensor(0.0))       
            
            
            # # # Sanity check
            # if len(collision_sample_idx) > 0: # visualize collision samples
            #     print(len(collision_sample_idx))
            #     k = collision_sample_idx[0]
            # if len(discomfort_c.nonzero()) > 0: # visualize discomfort samples
            #     # discomfort_sample_idx = torch.where(discomfort_c>0)
            #     # k = discomfort_sample_idx[0][0]
                
            #     # TODO: Compute the current time step's label and sensor grid for visualization
            #     print(next_label_grid[k][0].shape, next_sensor_grid[k].shape, transferred_next_sensor_grid[k].shape, disambig_R_map_next[k].shape, decoded[0,0].shape, self.curr_label_grid.shape, self.curr_sensor_grid.shape)
            # for k in range(5): #torch.where(disambig_c < 0)[0][:1]:
            #     curr_map_x, curr_map_y = self.curr_map_xy
            #     vis_grids = [[next_x_local[k], next_y_local[k], next_label_grid[k][0], "Next label grid"]]
            #     vis_grids = [[next_x_local[k], next_y_local[k], next_label_grid[k][1], "Next id grid"]]
            #     vis_grids.append([next_x_local[k], next_y_local[k], next_sensor_grid[k], "Next sensor grid"])
            #     vis_grids.append([curr_map_x, curr_map_y, transferred_next_sensor_grid[k], "Transferred next sensor grid"])
            #     vis_grids.append([next_x_local[k], next_y_local[k], collision_cost_grid[k], "Collision cost grid"])
            #     vis_grids.append([next_x_local[k], next_y_local[k], discomfort_cost_grid[k], "Discomfort cost grid"])
            #     # vis_grids.append([curr_map_x, curr_map_y, disambig_W_map_next[k], "Next disambig W map"])
            #     vis_grids.append([curr_map_x, curr_map_y, disambig_R_map_next[k], "Next disambig R map"])
            #     vis_grids.append([curr_map_x, curr_map_y, decoded[0,0], " Decoded grid at t=0"])
            #     vis_grids.append([curr_map_x, curr_map_y, self.curr_label_grid, "Current label grid"])
            #     vis_grids.append([curr_map_x, curr_map_y, self.curr_sensor_grid, "Current sensor grid"])
                
            #     # vis_grids.append([curr_map_x, curr_map_y, decoded[0,k], "PaS inference grid at t=0"])
            #     # # vis_grids.append([next_x_local[k], next_y_local[k], next_pas_inf_grid[k], "PaS inference grid at t=k"])
            #     # # To debug the Transfer_to_EgoGrid
            #     # empty_grid2 = torch.zeros_like(sensor_grid)
            #     # pas_inf_grid = Transfer_to_EgoGrid(map_xy, empty_grid2, self.curr_map_xy, self.curr_sensor_grid.unsqueeze(1), grid_res) # decoded in current time (t=0) 
            #     # vis_grids.append([next_x_local[k], next_y_local[k], pas_inf_grid[k], "Current sensor grid at t=k"])
            #     self.visualize(vis_grids, self.r_goal, disambig_c[k], r_state[0,k], next_r_state[0,k], next_h_state[0,k])
            # self.visualize_traj(r_state, next_r_state, self.r_goal)
        else:
            self.disambig_R_map = self.curr_sensor_grid.repeat(next_r_state.shape[1],1,1)
            disambig_c = torch.tensor(0.0).to(self.device)
            # self.visualize_traj(r_state, next_r_state, self.r_goal)
        
        ## Check if there were any collisions                
        # goal_reaching_c = 0 or -5
        # potential_c: -0.99~ 0.97
        # collision_c: [0, 10, 30, 70,110.,200.]  -> make this regardless of the number of occupancy cells. Now 0 or 10.
        # discomfort_c: [0., 10, 30, 70.] -> make this regardless of the number of occupancy cells. No 0 or 2.
        # disambig_c: 0 or [-8.3, ]
        cost = goal_reaching_c + potential_c + collision_c + discomfort_c + disambig_c
        
        # if torch.any(disambig_c != 0) or torch.any(discomfort_c > 0) or torch.any(collision_c>0):
        #     mask = torch.where(torch.logical_or(discomfort_c>0, collision_c>0))
        #     print('goal',goal_reaching_c[mask], 'potent',potential_c[mask], 'colli',collision_c[mask], 'discomfort',discomfort_c[mask], 'disambig', disambig_c[mask])
        self.next_map_xy = next_map_xy
        self.next_sensor_grid = next_sensor_grid
        return cost
    
    def plan(self, obs, decoded, config):
        """
        obs: full sequence (batch, sequence+lookahead_steps, 1+human_num, 7)
        decoded
        !! Not supporting batch yet.
        
        return: trajectory (1+human_num, (lookahead_steps)*4)
        """

        sequence = config.pas.sequence
        state = obs['mppi_vector'][0,sequence-1, 0].clone() # (1, 7) robot's current state
        ob = obs['mppi_vector'][0,sequence-1:, 1:].clone() # Using vector states (1+lookahead_steps, human_num, 7) human's current and future states
        self.curr_label_grid = obs['label_grid'][0, 0].clone() # (1, 2,  H, W)
        self.curr_sensor_grid = obs['grid'][0, -1].clone() # (1,sequence, H, W)
        self.curr_map_xy = obs['grid_xy'][0].clone()
        vis_ids = torch.unique(obs['vis_ids'][0]).clone()
        # print('all vis_ids', vis_ids)
        if len(vis_ids) > 1:
            self.vis_ids = vis_ids[1:] # remove dummy id (-1)
        else:
            self.vis_ids = torch.tensor([]).to(self.device)
            
        # ob = deepcopy(obs['grid'][0,sequence-1:]) # Using grid states
        self.r_goal = state[4:6]
        propagated_states, action = self.mppi_gym.command(state, ob, decoded)
        
        ## Copy obs['mppi_vector'] and assign the propated_states to the robot's states
        final_traj = obs['mppi_vector'][:, sequence-1:, :, :4].clone()
        final_traj[0,1:,0, :2] = propagated_states
        final_traj[0,1:,0, 2:] = action # velocity
                
        final_traj = final_traj.permute(0,2,1,3).reshape(-1,1)
        
        return final_traj, self.disambig_R_map
        
        
        
            
        
        
        
        
        
        
        
    
