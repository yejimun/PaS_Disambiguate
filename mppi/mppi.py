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
# TODO: improve transfer_to_ego_grid to be more efficient.


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

    def __init__(self, dynamics, running_cost, nx, noise_sigma, num_samples=100, horizon=30, device="cpu",
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
    def _running_cost(self, state, ob, decoded, u, t):
        return self.running_cost(state, ob, decoded, u, t)# if self.step_dependency else self.running_cost(state, u)

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
            state = self._dynamics(state[:2], action[a])
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
        state = state.repeat(self.M, 1, 1)[:, :, :2] # repeated current state # TODO: not using self.M
        
        
        # (batch, K, pred_horizon, human_num, 7) # repeated obs with pred_horizon
        ob = self.ob.unsqueeze(0).unsqueeze(0).repeat(self.M, K, 1, 1, 1) # vector obs

                
        states = []
        actions = []
        for t in range(T):
            u = self.u_scale * perturbed_actions[:, t].repeat(self.M, 1, 1)
            next_state = self._dynamics(state, u)
            next_state = self._sample_specific_dynamics(next_state, state, u, t)
            state = next_state
            c = self._running_cost(state, ob, self.decoded, u, t) 
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
        if self.u_max is not None:
            return torch.max(torch.min(action, self.u_max), self.u_min)
        return action

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


class SMPPI(MPPI):
    """Smooth MPPI by lifting the control space and penalizing the change in action from
    https://arxiv.org/pdf/2112.09988
    """

    def __init__(self, *args, w_action_seq_cost=1., delta_t=1., U_init=None, action_min=None, action_max=None,
                 **kwargs):
        self.w_action_seq_cost = w_action_seq_cost
        self.delta_t = delta_t

        super().__init__(*args, U_init=U_init, **kwargs)

        # these are the actual commanded actions, which is now no longer directly sampled
        self.action_min = action_min
        self.action_max = action_max
        if self.action_min is not None and self.action_max is None:
            if not torch.is_tensor(self.action_min):
                self.action_min = torch.tensor(self.action_min)
            self.action_max = -self.action_min
        if self.action_max is not None and self.action_min is None:
            if not torch.is_tensor(self.action_max):
                self.action_max = torch.tensor(self.action_max)
            self.action_min = -self.action_max
        if self.action_min is not None:
            self.action_min = self.action_min.to(device=self.d)
            self.action_max = self.action_max.to(device=self.d)

        # this smooth formulation works better if control starts from 0
        if U_init is None:
            self.action_sequence = torch.zeros_like(self.U)
        else:
            self.action_sequence = U_init
        self.U = torch.zeros_like(self.U)

    def get_params(self):
        return f"{super().get_params()} w={self.w_action_seq_cost} t={self.delta_t}"

    def shift_nominal_trajectory(self):
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init
        self.action_sequence = torch.roll(self.action_sequence, -1, dims=0)
        self.action_sequence[-1] = self.action_sequence[-2]  # add T-1 action to T

    def get_action_sequence(self):
        return self.action_sequence

    def reset(self):
        self.U = torch.zeros_like(self.U)
        self.action_sequence = torch.zeros_like(self.U)

    def change_horizon(self, horizon):
        if horizon < self.U.shape[0]:
            # truncate trajectory
            self.U = self.U[:horizon]
            self.action_sequence = self.action_sequence[:horizon]
        elif horizon > self.U.shape[0]:
            # extend with u_init
            extend_for = horizon - self.U.shape[0]
            self.U = torch.cat((self.U, self.u_init.repeat(extend_for, 1)))
            self.action_sequence = torch.cat((self.action_sequence, self.action_sequence[-1].repeat(extend_for, 1)))
        self.T = horizon

    def _bound_d_action(self, control):
        if self.u_max is not None:
            return torch.max(torch.min(control, self.u_max), self.u_min)  # action
        return control

    def _bound_action(self, action):
        if self.action_max is not None:
            return torch.max(torch.min(action, self.action_max), self.action_min)
        return action

    def _command(self, state, ob, decoded):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        self.ob = ob.to(dtype=self.dtype, device=self.d)
        self.decoded = decoded.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()

        self._compute_weighting(cost_total)
        perturbations = torch.sum(self.omega.view(-1, 1, 1) * self.noise, dim=0)

        self.U = self.U + perturbations
        # U is now the lifted control space, so we integrate it
        self.action_sequence += self.U * self.delta_t

        action = self.get_action_sequence()[:self.u_per_command]
        # reduce dimensionality if we only need the first command
        if self.u_per_command == 1:
            action = action[0]
        return action

    def _compute_perturbed_action_and_noise(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.T))
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_control = self.U + noise
        # naively bound control
        self.perturbed_control = self._bound_d_action(perturbed_control)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.perturbed_action = self.action_sequence + perturbed_control * self.delta_t
        self.perturbed_action = self._sample_specific_actions(self.perturbed_action)
        self.perturbed_action = self._bound_action(self.perturbed_action)

        self.noise = (self.perturbed_action - self.action_sequence) / self.delta_t - self.U

    def _compute_total_cost_batch(self):
        self._compute_perturbed_action_and_noise()
        if self.noise_abs_cost:
            action_cost = self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv
            # NOTE: The original paper does self.lambda_ * torch.abs(self.noise) @ self.noise_sigma_inv, but this biases
            # the actions with low noise if all states have the same cost. With abs(noise) we prefer actions close to the
            # nomial trajectory.
        else:
            action_cost = self.lambda_ * self.noise @ self.noise_sigma_inv  # Like original paper

        # action difference as cost
        action_diff = self.u_scale * torch.diff(self.perturbed_action, dim=-2)
        action_smoothness_cost = torch.sum(torch.square(action_diff), dim=(1, 2))
        # handle non-homogeneous action sequence cost
        action_smoothness_cost *= self.w_action_seq_cost

        rollout_cost, self.states, actions = self._compute_rollout_costs(self.perturbed_action)
        self.actions = actions / self.u_scale

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))
        self.cost_total = rollout_cost + perturbation_cost + action_smoothness_cost
        return self.cost_total


class TimeKernel:
    """Kernel acting on the time dimension of trajectories for use in interpolation and smoothing"""

    def __call__(self, t, tk):
        raise NotImplementedError


class RBFKernel(TimeKernel):
    def __init__(self, sigma=1):
        self.sigma = sigma

    def __repr__(self):
        return f"RBFKernel(sigma={self.sigma})"

    def __call__(self, t, tk):
        d = torch.sum((t[:, None] - tk) ** 2, dim=-1)
        k = torch.exp(-d / (1e-8 + 2 * self.sigma ** 2))
        return k


class KMPPI(MPPI):
    """MPPI with kernel interpolation of control points for smoothing"""

    def __init__(self, *args, num_support_pts=None, kernel: TimeKernel = RBFKernel(), **kwargs):
        super().__init__(*args, **kwargs)
        self.num_support_pts = num_support_pts or self.T // 2
        # control points to be sampled
        self.theta = torch.zeros((self.num_support_pts, self.nu), dtype=self.dtype, device=self.d)
        self.Tk = None
        self.Hs = None
        # interpolation kernel
        self.interpolation_kernel = kernel
        self.intp_krnl = None
        self.prepare_vmap_interpolation()

    def get_params(self):
        return f"{super().get_params()} num_support_pts={self.num_support_pts} kernel={self.interpolation_kernel}"

    def reset(self):
        super().reset()
        self.theta.zero_()

    def shift_nominal_trajectory(self):
        super().shift_nominal_trajectory()
        self.theta, _ = self.do_kernel_interpolation(self.Tk[0] + 1, self.Tk[0], self.theta)

    def get_action_sequence(self):
        return self.action_sequence
    
    def do_kernel_interpolation(self, t, tk, c):
        K = self.interpolation_kernel(t.unsqueeze(-1), tk.unsqueeze(-1))
        Ktktk = self.interpolation_kernel(tk.unsqueeze(-1), tk.unsqueeze(-1))
        # print(K.shape, Ktktk.shape)
        # row normalize K
        # K = K / K.sum(dim=1).unsqueeze(1)

        # KK = K @ torch.inverse(Ktktk)
        KK = torch.linalg.solve(Ktktk, K, left=False)

        return torch.matmul(KK, c), K

    def prepare_vmap_interpolation(self):
        self.Tk = torch.linspace(0, self.T - 1, int(self.num_support_pts), device=self.d, dtype=self.dtype).unsqueeze(
            0).repeat(self.K, 1)
        self.Hs = torch.linspace(0, self.T - 1, int(self.T), device=self.d, dtype=self.dtype).unsqueeze(0).repeat(
            self.K, 1)
        self.intp_krnl = vmap(self.do_kernel_interpolation)

    def deparameterize_to_trajectory_single(self, theta):
        return self.do_kernel_interpolation(self.Hs[0], self.Tk[0], theta)

    def deparameterize_to_trajectory_batch(self, theta):
        assert theta.shape == (self.K, self.num_support_pts, self.nu)
        return self.intp_krnl(self.Hs, self.Tk, theta)

    def _compute_perturbed_action_and_noise(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action
        noise = self.noise_dist.rsample((self.K, self.num_support_pts))
        perturbed_control_pts = self.theta + noise
        # control points in the same space as control and should be bounded
        perturbed_control_pts = self._bound_action(perturbed_control_pts)
        self.noise_theta = perturbed_control_pts - self.theta
        perturbed_action, _ = self.deparameterize_to_trajectory_batch(perturbed_control_pts)
        perturbed_action = self._sample_specific_actions(perturbed_action)
        # naively bound control
        self.perturbed_action = self._bound_action(perturbed_action)
        # bounded noise after bounding (some got cut off, so we don't penalize that in action cost)
        self.noise = self.perturbed_action - self.U

    def _command(self, state, ob, decoded):
        if not torch.is_tensor(state):
            state = torch.tensor(state)
        self.state = state.to(dtype=self.dtype, device=self.d)
        self.ob = ob.to(dtype=self.dtype, device=self.d)
        self.decoded = decoded.to(dtype=self.dtype, device=self.d)
        cost_total = self._compute_total_cost_batch()

        self._compute_weighting(cost_total)
        perturbations = torch.sum(self.omega.view(-1, 1, 1) * self.noise_theta, dim=0)

        self.theta = self.theta + perturbations
        self.U, _ = self.deparameterize_to_trajectory_single(self.theta)
        action = self.U
        
        # action = self.get_action_sequence()[:self.u_per_command]
        # # reduce dimensionality if we only need the first command
        # if self.u_per_command == 1:
        #     action = action[0]
        propagated_states = []
        ## Rollouts with the final action
        for a in range(self.u_per_command):
            state = self._dynamics(state[:2], action[a])
            propagated_states.append(state)
            
        propagated_states = torch.stack(propagated_states)
        return propagated_states, action



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
        noise_sigma = torch.eye(nx, device=self.device, dtype=torch.double)*2.
        N_SAMPLES = self.config.diffstack.num_samples
        TIMESTEPS = self.lookahead_steps
        lambda_ = self.config.diffstack.lambda_
        ACTION_LOW = -self.config.robot.v_pref # Not used
        ACTION_HIGH = self.config.robot.v_pref
        
        self.mppi_gym = MPPI(self.dynamics, self.running_cost, nx, noise_sigma, num_samples=N_SAMPLES, horizon=TIMESTEPS,
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
        next_r_state = state + action * self.time_step
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
        ax.plot(r_state[0,0,0].cpu(), r_state[0,0,1].cpu(), 'bo', markersize=10)
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
        

    def running_cost(self, state, ob, decoded, action, t):
        """
        state: robot's state at t (batch, K, 2). Note that this is not the current state but a rolled state.
        ob: human's state at curr_t:curr_t+look_ahead (batch, K, 1+lookahead_steps, human_num, 7) 
        """
        r_state = state
        r_radius = torch.tensor(self.config.robot.radius, device=self.device).repeat(r_state.shape[0],r_state.shape[1]) # (batch, K)
        # self.r_goal = state[:, 4:6]
        # next_r_goal = self.r_goal #state[cur_t+1, 0, 4:6] # should be the same as r_goal in our case
        r_goal = torch.tensor(self.r_goal, device=self.device).repeat(r_state.shape[0],r_state.shape[1],1) # (batch, K)
        next_r_state = self.dynamics(state, action)#.unsqueeze(-2)
        
        ########### Costs based on ego's state vector ###########
        ## reaching goal
        goal_reaching_c = torch.tensor(0.0).repeat(r_state.shape[1])
        goal_reaching_c = torch.where(torch.linalg.norm(next_r_state - r_goal, axis=-1) < r_radius,
                                      torch.tensor(self.config.reward.success_reward),
                                      torch.tensor(0.0)).view(-1)
        
        ## moving towards the goal
        potential_cur = torch.linalg.norm(r_state - r_goal, axis=-1)
        potential_next = torch.linalg.norm(next_r_state - r_goal, axis=-1)
        potential_c = torch.tensor(2.) * (potential_next-potential_cur) # (batch, K) # penalty if moving away from the goal (potential increases)
        # No potential cost if the robot is already at the goal
        potential_c = torch.where(potential_cur < r_radius, torch.tensor(0.0), potential_c).view(-1)
    
        
        ########### Costs based on ego's observation OGM ###########
        # Collision cost (in the next state) 
        collision_penalty = self.config.reward.collision_penalty
        next_h_state = ob[:, :, t+1, :, :2]
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
      
            
            # # Sanity check
            # if len(collision_sample_idx) > 0: # visualize collision samples
                # print(len(collision_sample_idx))
                # k = collision_sample_idx[0]
            # if len(discomfort_c.nonzero()) > 0: # visualize discomfort samples
            #     # discomfort_sample_idx = torch.where(discomfort_c>0)
            #     # k = discomfort_sample_idx[0][0]
                
            #     # TODO: Compute the current time step's label and sensor grid for visualization
                # print(next_label_grid[k][0].shape, next_sensor_grid[k].shape, transferred_next_sensor_grid[k].shape, disambig_R_map_next[k].shape, decoded[0,0].shape, self.curr_label_grid.shape, self.curr_sensor_grid.shape)
            # for k in torch.where(disambig_c < 0)[0][:1]:
            #     curr_map_x, curr_map_y = self.curr_map_xy
            #     # vis_grids = [[next_x_local[k], next_y_local[k], next_label_grid[k][0], "Next label grid"]]
            #     vis_grids = [[next_x_local[k], next_y_local[k], next_label_grid[k][1], "Next id grid"]]
            #     vis_grids.append([next_x_local[k], next_y_local[k], next_sensor_grid[k], "Next sensor grid"])
            #     # vis_grids.append([curr_map_x, curr_map_y, transferred_next_sensor_grid[k], "Transferred next sensor grid"])
            #     # vis_grids.append([next_x_local[k], next_y_local[k], collision_cost_grid[k], "Collision cost grid"])
            #     # vis_grids.append([next_x_local[k], next_y_local[k], discomfort_cost_grid[k], "Discomfort cost grid"])
            #     # vis_grids.append([curr_map_x, curr_map_y, disambig_W_map_next[k], "Next disambig W map"])
            #     # vis_grids.append([curr_map_x, curr_map_y, disambig_R_map_next[k], "Next disambig R map"])
            #     # vis_grids.append([curr_map_x, curr_map_y, decoded[0,0], " Decoded grid at t=0"])
            #     # vis_grids.append([curr_map_x, curr_map_y, self.curr_label_grid, "Current label grid"])
            #     # vis_grids.append([curr_map_x, curr_map_y, self.curr_sensor_grid, "Current sensor grid"])
                
            #     # vis_grids.append([curr_map_x, curr_map_y, decoded[0,k], "PaS inference grid at t=0"])
            #     # vis_grids.append([next_x_local[k], next_y_local[k], next_pas_inf_grid[k], "PaS inference grid at t=k"])
            #     # To debug the Transfer_to_EgoGrid
            #     # empty_grid2 = torch.zeros_like(sensor_grid)
            #     # pas_inf_grid = Transfer_to_EgoGrid(torch.stack(self.curr_map_xy).permute(1,0,2,3)[0], self.curr_sensor_grid[0], map_xy, empty_grid2, grid_res) # decoded in current time (t=0) 
            #     # vis_grids.append([next_x_local[k], next_y_local[k], pas_inf_grid[k], "Current sensor grid at t=k"])
            #     self.visualize(vis_grids, self.r_goal, disambig_c[k])
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
        cost = goal_reaching_c.float() + potential_c + collision_c.float() + discomfort_c.float() + disambig_c.float()
        
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
    
    def plan(self, obs, decoded, config):
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
        self.curr_label_grid = obs['label_grid'][0, 0].clone() # (1, 2,  H, W)
        self.curr_sensor_grid = obs['grid'][0, -1].clone()  # (1,sequence, H, W)
        self.curr_map_xy = obs['grid_xy'][0].clone() 
        vis_ids = torch.unique(obs['vis_ids'][0]).clone() 
        if len(vis_ids) > 1:
            self.vis_ids = vis_ids[1:] # remove dummy id (-1)
        else:
            self.vis_ids = torch.tensor([]).to(self.device)
            
        self.r_goal = state[4:6]# (batch, 2)
        propagated_states, action = self.mppi_gym.command(state, ob, decoded)
        
        ## Copy obs['mppi_vector'] and assign the propated_states to the robot's states
        final_traj = obs['mppi_vector'][:, sequence-1:, :, :4].clone() # (1, 1+lookahead_steps, 1+human_num, 4)
        final_traj[0,1:,0, :2] = propagated_states # (lookahead_steps, 2)
        final_traj[0,1:,0, 2:] = action # velocity # (lookahead_steps, 2)
                
        final_traj = final_traj.permute(0,2,1,3).reshape(-1,1) # (1, 1+human_num, 1+lookahead_steps, 4) 
        return final_traj, self.disambig_R_map
        
        
        
            
        
        
        
        
        
        
        
    
