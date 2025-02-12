import gym
import numpy as np
from numpy.linalg import norm
import copy
from crowd_sim.envs.utils.action import ActionRot, ActionXY
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.utils.human import Human

from crowd_sim.envs import CrowdSim
from crowd_sim.envs.generateLabelGrid import generateLabelGrid
from crowd_sim.envs.generateSensorGrid import generateSensorGrid
from crowd_sim.envs.grid_utils import global_grid

import torch
import glob
import imageio
import os
import math
from collections import deque
from copy import deepcopy
from scipy.spatial import cKDTree
import pdb


class CrowdSimDict(CrowdSim):
    def __init__(self):
        """
        Movement simulation for n+1 agents
        Agent can either be human or robot.
        humans are controlled by a unknown and fixed policy.
        robot is controlled by a known and learnable policy.
        """
        super().__init__()
        self.desiredVelocity=[0.0,0.0]
        self.robot_states_copy = None
        self.human_states_copy = None
        self.curr_map_xy = []
        self.curr_sensor_grid = []
        self.prev_disambig_reward = 0
        self.disambig_reward = 0
        self.disambig_reward_grid = []

    def set_robot(self, robot):
        self.robot = robot
        """[summary]
        """
        if self.collectingdata:   
            robot_vec_length = 9
        else:
            if self.config.action_space.kinematics=="holonomic":
                robot_vec_length = 4
            else:
                robot_vec_length = 5
        d={}         
        if self.collectingdata:    
            vec_length = robot_vec_length+5*self.human_num
            d['vector'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, vec_length,), dtype = np.float32)                    
            d['label_grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, *self.grid_shape), dtype = np.float32) 
            d['sensor_grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, *self.grid_shape), dtype = np.float32) 
            
        else:
            if self.config.robot.policy =='pas_rnn' or self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
                d['vector'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, robot_vec_length,), dtype = np.float32)  
            if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
                full_sequence = self.config.pas.sequence + self.config.diffstack.lookahead_steps
                d['mppi_vector'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(full_sequence, 1+self.human_num, 7), dtype = np.float32)
            ## TODO: make the grid shape to sequence + future once the fast ray tracing is implemented              
            if self.config.pas.seq_flag or self.config.pas.encoder_type != 'cnn' :
                d['grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.pas.sequence, *self.grid_shape), dtype = np.float32) 
                d['label_grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, *self.grid_shape), dtype = np.float32) 
            else:
                d['grid'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1, *self.grid_shape), dtype = np.float32)
            d['vis_ids'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(self.config.pas.sequence, self.human_num,), dtype = np.float32)
            d['grid_xy'] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(2, *self.grid_shape), dtype = np.float32)
        self.observation_space=gym.spaces.Dict(d)

        if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
            high = np.inf * np.ones([1*(self.config.diffstack.lookahead_steps+1)*4, ])
            ## Uncomment for the "fan" planner
            # high = np.inf * np.ones([700*(self.config.diffstack.lookahead_steps+1)*4, ])
            self.action_space = gym.spaces.Box(-high, high,dtype=np.float32)
        else:
            high = np.inf * np.ones([2, ])
            self.action_space = gym.spaces.Box(-high, high, dtype=np.float32)
        
    def copy_human(self, human):
        new_human = Human(self.config, 'humans')
        new_human.set(human.px, human.py, human.gx, human.gy, human.vx, human.vy, human.theta, human.radius, human.v_pref)
        return new_human
    
    def copy_robot(self, robot):
        from crowd_sim.envs.utils.robot import Robot
        new_robot = Robot(self.config, 'robot')
        new_robot.set(robot.px, robot.py, robot.gx, robot.gy, 0., 0., robot.theta, robot.radius, 0.)
        return new_robot
    
    
    def obsV_robot_human(self, human_visibility):
        obsV = [self.robot.get_diff_state()]
        for i, human in enumerate(self.humans):
            if human_visibility[i]:
                obsV.append(human.get_diff_state())
            else:
                obsV.append(self.dummy_human.get_diff_state())
        return obsV
        

    def generate_ob(self, reset):
        ob = {}
        ego_id = 100 # robot_id
        self.dict_update()
        ego_dict, other_dict = self.ego_other_dict(ego_id)
        
        label_grid, x_map, y_map = generateLabelGrid(ego_dict, other_dict, res=self.grid_res)
        map_xy = [x_map, y_map]

        if self.gridsensor == 'sensor' or self.collectingdata:
            visible_id, sensor_grid = generateSensorGrid(label_grid, ego_dict, other_dict, map_xy, self.FOV_radius, res=self.grid_res)
            self.visible_ids.append(visible_id)
        else:
            visible_id = np.unique(label_grid[1])[:-1]
            self.visible_ids.append(visible_id)
        
        
        # human_visibility = [True if i in self.visible_ids else False for i in range(self.human_num)] 
        dummy_human_visibility = [True for i in range(self.human_num)] # To collect the gt positions of humans            
        self.xy_local_grid.append([x_map, y_map])                
        
        self.update_last_human_states(dummy_human_visibility, reset=reset)
        
        # ########### for diffstack and mppi ################
        obsV = self.obsV_robot_human(dummy_human_visibility)
        if reset:
            # copy the current state for the past sequence length
            # attach the current state to the end of the list
            for i in range(self.config.pas.sequence-1):
                self.states_list.append(obsV)
            # print(np.array(self.states_list)[:, :, :2])
                        
        # get_diff_state(): [self.px, self.py, self.theta, self.v, self.gx, self.gy, self.radius]
        self.states_list.append(obsV)
        
        ## Update all agents for a lookahead step, but don't update the global states
        # TODO: Create sensor grid for the next step
        # if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
        # copy robot and human agents
        robot_copy = self.copy_robot(self.robot)
        humans_copy = []
        for human in self.humans:
            humans_copy.append(self.copy_human(human))
            
        # update robot and human agents
        lookahead_states = self.lookahead_step(robot_copy, humans_copy, self.config.diffstack.lookahead_steps)
            
        self.lookahead_list.append(lookahead_states)
           
        #####################################
        # if self.config.robot.policy =='pas_rnn' or self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
        if self.config.pas.gridtype == 'global' or self.collectingdata:
            ob['vector'] = self.robot.get_full_state_list()
        elif self.config.pas.gridtype == 'local':
            # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
            robot_state_np = np.array(self.robot.get_full_state_list()).copy()
            if self.robot.kinematics == 'holonomic':               
                robot_vector = np.zeros(4)
                robot_vector[:2] = robot_state_np[:2]-robot_state_np[5:7]
                robot_vector[2:4] = robot_state_np[2:4]
                ob['vector'] = robot_vector  # (px-gx, py-gy, vx, vy)     
            else: 
                robot_vector = np.zeros(5) 
                robot_vector[:5] = robot_state_np[:5]
                robot_vector[:2] = deepcopy(robot_state_np[:2]-robot_state_np[5:7])
                robot_vector[2] = robot_state_np[-1]                    
                ob['vector'] = robot_vector # unicycle # (px-gx, py-gy, theta, v, w)
            ob['mppi_vector'] = np.array(self.states_list[-self.config.pas.sequence:] + lookahead_states) # (sequence+lookahead_steps,1+n_humans, 7) )
            # print(ob['mppi_vector'][:, :, :2])
        # if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
        #     # ob['vector'] should have past sequence length from self.states and future sequence length from the lookahead step
        #     # stack per agent
        #     # print(np.array(self.states_list[-self.config.pas.sequence:]).shape, np.array(lookahead_states).shape, np.array(self.states[-self.config.pas.sequence:] + lookahead_states).shape)            
            
        #     if reset:
        #         print(reset)
        #         # copy the current state for the past sequence length
        #         # attach the current state to the end of the list
        #         for i in range(self.config.pas.sequence):
        #             obsV = robot_vector + 
        #             self.states_list.append(obsV)
        #         print(np.array(obsV)[:, :2])
        
        
        #     ob['mppi_vector'] = np.array(self.states_list[-self.config.pas.sequence:] + lookahead_states) # (sequence+lookahead_steps,1+n_humans, 7) )
        #     # print(len(self.states_list[-self.config.pas.sequence:]), np.array(self.states_list[-self.config.pas.sequence:])[:, :, :2])
        #     ## update the velocity to be the next velocity (s_{t-1}, a_{t})
        #     # print(ob['vector'].shape, np.array(self.states_list[-self.config.pas.sequence-1:]).shape)
            
        #     # ob['vector'][:,:,3:5] = np.array(self.states_list[-self.config.pas.sequence-1:] + lookahead_states)[-len(ob['vector'])-1:-1,:,3:5] # (sequence+lookahead_steps,1+n_humans, 7) )

        # else:
        #     raise NotImplementedError                
        
        
        # For collecting data
        if self.collectingdata:
            # When robot is executed by ORCA
            for i, human in enumerate(self.humans):                    
                # observation for robot. Don't include robot's state
                self.ob = [] 
                for other_human in self.humans:
                    if other_human != human:
                        # Chance for one human to be blind to some other humans
                        if self.random_unobservability and i == 0:
                            if np.random.random() <= self.unobservable_chance or not self.detect_visible(human,
                                                                                                        other_human):
                                self.ob.append(self.dummy_human.get_observable_state())
                            else:
                                self.ob.append(other_human.get_observable_state())
                        # Else detectable humans are always observable to each other
                        elif self.detect_visible(human, other_human):
                            self.ob.append(other_human.get_observable_state())
                        else:
                            self.ob.append(self.dummy_human.get_observable_state())
            # TODO: !! human visibility should be True for all humans to use the gt positions. But not a big deal since it was only used for sanity check.
            ob['vector'].extend(list(np.ravel(self.last_human_states))) # add human states to vector
            ob['grid_xy'] = map_xy
            ob['label_grid'] = label_grid # both the label grid and id grid
            ob['sensor_grid'] = sensor_grid              
        else:    
            if self.gridsensor == 'sensor':
                if self.config.pas.encoder_type != 'cnn' :
                    self.sequence_grid.append(sensor_grid)
                    if len(self.sequence_grid) < self.config.pas.sequence:
                        gd = deepcopy(self.sequence_grid)
                        gd1 = np.stack([gd[0] for i in range(self.config.pas.sequence-len(self.sequence_grid))])
                        stacked_grid = deepcopy(np.concatenate([gd1, gd]))
                    else:
                        stacked_grid = deepcopy(np.stack(self.sequence_grid))
                
                    # print('visible_id', visible_id)
                    cur_vis_grids = np.ones(self.human_num) * -1
                    if len(visible_id) > 0:
                        for h_id in visible_id:
                            cur_vis_grids[int(h_id)] = h_id
                    self.vis_ids.append(cur_vis_grids)
                    if len(self.vis_ids) < self.config.pas.sequence:
                        dummy_vis_grid = np.stack([self.vis_ids[0] for i in range(self.config.pas.sequence-len(self.vis_ids))])                        
                        ob['vis_ids'] = np.concatenate([dummy_vis_grid, np.stack(self.vis_ids)])
                    else:
                        ob['vis_ids'] = np.stack(self.vis_ids)[-self.config.pas.sequence:]
                    # print(self.vis_ids)
                    ob['label_grid'] = label_grid
                    ob['grid'] = stacked_grid
                else:
                    ob['label_grid'] = label_grid
                    ob['grid'] = sensor_grid
            elif self.gridsensor == 'gt' :
                if self.config.pas.seq_flag or self.config.pas.encoder_type != 'cnn' :
                    self.sequence_grid.append(label_grid[0])
                    if len(self.sequence_grid) < self.config.pas.sequence:
                        gd = deepcopy(self.sequence_grid)
                        gd1 = np.stack([gd[0] for i in range(self.config.pas.sequence-len(self.sequence_grid))])
                        stacked_grid = deepcopy(np.concatenate([gd1, gd]))
                    else:
                        stacked_grid = deepcopy(np.stack(self.sequence_grid))                            
                    ob['grid'] = stacked_grid   
                    ob['label_grid'] = label_grid                 
                    
                else:
                    ob['grid'] = label_grid[0]
                    
            ob['grid_xy'] = map_xy

        # if self.disambig_reward_flag:  
        #     self.prev_disambig_reward = self.disambig_reward
        #     self.disambig_reward = self.calculate_disambig_reward(sensor_grid, method=self.config.reward.disambig_method)
        
        # To visualize uncertainty reward grid, (i) uncomment the following line & self.disambig_reward_grid.append(disambig_reward_map) in calculate_disambig_reward()
        # (ii) replace ob['grid'] with sensor_grid in the upper line.
        # (iii) commment out the line ob['sensor_grid'] = sensor_grid
        # (iv) change the setting to collecting data
        
        # ob['sensor_grid'] = np.array([self.disambig_reward_grid[-1]])
        self.states.append([self.robot.get_full_state(), [human.get_full_state() for human in self.humans]])
        return ob



    def dict_update(self,):
        """[summary]
        Updates the current state dictionary (self.robot_dict, self.human_dict)
        For creating label/sensor grid
        """
        human_id = []
        human_pos = []
        human_v = []
        human_a = []
        human_radius = []
        human_theta = []
        human_goal = []

        for i, human in enumerate(self.humans):
            theta = human.theta 
            human_v.append([human.vx, human.vy])
            human_pos.append([human.px, human.py])
            human_radius.append(human.radius)
            human_goal.append([human.gx, human.gy])
            human_a.append([human.ax, human.ay])
            human_id.append(i)
            human_theta.append(theta)

        robot_pos = np.array([self.robot.px,self.robot.py])
        robot_theta = self.robot.theta
        robot_radius = self.robot.radius
        robot_v = np.array([self.robot.vx, self.robot.vy])
        robot_goal = np.array([self.robot.gx, self.robot.gy])
        robot_a = np.array([self.robot.ax, self.robot.ay])

        keys = ['id','pos', 'v', 'a', 'r', 'theta', 'goal']
        self.robot_values = [100, robot_pos, robot_v, robot_a, robot_radius, robot_theta, robot_goal]
        self.robot_dict = dict(zip(keys, self.robot_values))

        self.human_values = [human_id, human_pos, human_v, human_a, human_radius, human_theta, human_goal]
        self.humans_dict = dict(zip(keys, self.human_values))


    def ego_other_dict(self, ego_id):
        """[summary]
        For creating label/sensor grid
        Ego can be either the robot or one of the pedestrians
        Args:
            ego_id (int): 100 for robot and 0~(n-1) for n pedestrians
        Returns:
            [type]: [description]
        """
        keys = ['id','pos', 'v', 'a', 'r', 'theta', 'goal']
        if ego_id == 100: # When the ego is robot
            ego_dict = self.robot_dict
            other_dict = self.humans_dict
        else: # When the ego is a pedestrian
            human_values = np.array(copy.copy(self.human_values) ,dtype=object)
            ego_dict = dict(zip(keys, human_values[:,ego_id].tolist()))

            other_values = np.delete(human_values, ego_id, 1)  # deleting the ego info from the pedestrians' info
            robot_values = np.reshape(self.robot_values, [other_values.shape[0],-1])
            other_values = np.hstack((other_values, robot_values)) # combining robot and other pedestrians info (for grid w/ robot)
            other_dict = dict(zip(keys, other_values))
        return ego_dict, other_dict

    def update_robot_goals(self,):
        ## Update robot goal from generated goal list to promote exploration during collecting data (policy: ORCA)
        step = int(self.global_time/self.time_step)
        self.robot.gx = self.robot_goals[step][0]
        self.robot.gy = self.robot_goals[step][1]

    def reset(self, ):
        """
        Set px, py, gx, gy, vx, vy, theta for robot and humans
        :return:
        """
        self.visible_ids = []
        self.Con = None
        self.cbar = None
        self.n_loop = 0

        if self.phase is not None:
            phase = self.phase

        if self.robot is None:
            raise AttributeError('robot has to be set!')
        assert phase in ['train', 'val', 'test']

        self.global_time = 0    

        self.all_videoGrids = []  # Store the FOV grid for rendering
        self.xy_local_grid = [] # Store the FOV grid coordinate for rendering

        if self.collectingdata and self.sim == 'crosswalk': 
            # During the data collection, episodes terminate when there's no reference agent (e.g. both reference agent reached their goals in crosswalk scenario)
            self.ref_goal_reaching = [False, False]
        
        self.sequence_grid = deque(maxlen=self.config.pas.sequence)
        # self.vis_ids = deque([np.ones(self.human_num)*(-1) for i in range(self.config.pas.sequence)], maxlen=self.config.pas.sequence)
        self.vis_ids = deque(maxlen=self.config.pas.sequence)
        self.states = list()
        self.states_list = list() # history + current
        self.lookahead_list = list()
        self.planned_traj = list()
        self.traj_candidates = list()
        self.desiredVelocity = [0.0, 0.0]
        self.humans = []
        # train, val, and test phase should start with different seed.
        # case capacity: the maximum number for train(max possible int -2000), val(1000), and test(1000)
        # val start from seed=0, test start from seed=case_capacity['val']=1000
        # train start from self.case_capacity['val'] + self.case_capacity['test']=2000
        counter_offset = {'train': self.case_capacity['val'] + self.case_capacity['test'],
                          'val': 0+3, 'test': self.case_capacity['val']}

        # here we use a counter to calculate seed. The seed=counter_offset + case_counter

        np.random.seed(counter_offset[phase] + self.case_counter[phase] + self.thisSeed)
        self.generate_robot_humans(phase)


        # If configured to randomize human policies, do so
        if self.random_policy_changing:
            self.randomize_human_policies()

        # case size is used to make sure that the case_counter is always between 0 and case_size[phase]
        self.case_counter[phase] = (self.case_counter[phase] + int(1*self.nenv)) % self.case_size[phase]


        # get robot observation
        ob = self.generate_ob(reset=True)
        if self.robot.kinematics == 'unicycle' and (self.config.robot.policy != 'pas_diffstack' or self.config.robot.policy != 'pas_mppi'):  # when 'pas_diffstack', action is returend as a part of the observation
            ob['vector'][3:5] =  np.array([0,0]) # TODO: needs to be updated to actual initial velocity
            
        # initialize current sensor grid and map_xy for disambiguation reward
        if self.config.reward.disambig_reward_flag and self.config.robot.policy == 'pas_rnn':
            self.curr_sensor_grid = deepcopy(ob['grid'][[-1]])#.reshape(1, *self.grid_shape)
            self.curr_map_xy = deepcopy(np.array(ob['grid_xy'])).reshape(2, *self.grid_shape)

        # initialize potential
        self.potential = - abs(np.linalg.norm(np.array([self.robot.px, self.robot.py]) - np.array([self.robot.gx, self.robot.gy])))
        return ob
    
    def pointinpolygon(self,x,y,poly):
        """
        inputs:
        - x: (K, H*W,) tensor of floats, where x[i] is the i-th x coordinate
        - y: (K, H*W,) tensor of floats, where y[i] is the i-th y coordinate
        - poly: (4, K, 2) tensor of floats, where poly[j] is the j-th vertex of the polygon
        return inside = (K, H*W) tensor of bools, where inside[i,j] is True if (x[i],y[i]) is inside poly[j]
        """
        n = len(poly) # (4,)
        K = len(x)
        HW = len(x[0])
        # Ensure the data types match
        poly = poly
        x = x#.to(poly.dtype)
        y = y#.to(poly.dtype)
        # poly = poly#.to(poly.dtype)
        
        inside = np.zeros((K, HW), dtype=bool) # (K, H*W)
        p2x = np.zeros((K,1))
        p2y = np.zeros((K,1))
        xints = np.zeros(x.shape, dtype=x.dtype)
        p1x,p1y = poly[0,:,[0]], poly[0,:,[1]] # (K,1), (K,1)
        
        for i in range(1,n+1):
            p2x,p2y = poly[i % n,:,[0]], poly[i % n,:,[1]] # (K,1), (K,1)
            mask = (y> np.min([p1y, p2y])) & (y <= np.max([p1y, p2y])) & (x <= np.max([p1x, p2x])) # (K, H*W)
            p_mask = (p1y != p2y)
            xint_mask = mask & p_mask  # (K, H*W)
            # xint_mask_flat = xint_mask.flatten()
            # y_flat = y.flatten()
            # xint_flat = xints.flatten()
            # # xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
            
            xints[xint_mask] = ((y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x)[xint_mask]
            # xint_flat[xint_mask_flat] = (y_flat[xint_mask_flat]-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
            # pdb.set_trace()
            inside_mask = mask & ((p1x==p2x) | (x <= xints))
            inside[inside_mask] = ~inside[inside_mask]
            
            p1x,p1y = p2x,p2y
        return inside
    
    def transfer_grid_data(self, curr_xy, curr_grids, next_xy, next_grids, distance_threshold=0.1, batch_chunk_size=1):
        """
        Transfers data from curr_grids to next_grids based on coordinate mapping
        from curr_xy (ego grid) to next_xy (map grid) only if the coordinates are within
        a specified distance threshold.
        
        
        Arguments:
        curr_xy = [[B,H,W], [B,H,W]]   
        curr_grids = [B,H,W] 
        next_xy = [[H,W], [H,W]]
        next_grids = [H,W]
        distance_threshold -- A float specifying the maximum distance for considering two points a match.

        Returns:
        Updated grids with the transferred data from next_grids.
        """
        # next_grids = next_grids.squeeze(0) # (1, batch_size, H, W) -> (batch_size, H, W)
        # curr_grids = curr_grids.squeeze(0) # (1, batch_size, H, W) -> (batch_size, H, W)
        B, H, W = curr_grids.shape
        
        # Initialize result tensor for updated pred_maps
        updated_grids = np.ones_like(curr_grids)*0.5
        
        next_xy_stacked = next_xy.transpose(1,2, 0)  # (H, W, 2)
        curr_xy_stacked = np.stack(curr_xy).transpose(1,2,0).reshape(B,H,W,2)  # (B, H, W, 2)
        # # Compute pairwise distances between next_xy and curr_xy
        # # (H*W, B, 2)-(H*W, 1, 2) = (H*W, B, 1)
        # pdb.set_trace()
        
        # dists = torch.cdist(curr_xy_stacked.view(B, -1,2).transpose(1,0,2),next_xy_stacked.view(-1, 2) .unsqueeze(1), p=2)  # (H*W, B, 1)
        next_xy_flat = next_xy_stacked.reshape(-1, 2)  # (H*W, 2)
        kdtree = cKDTree(next_xy_flat)
            
        # Process the batches in chunks to avoid memory overflow
        for start in range(0, B, batch_chunk_size):
            end = min(start + batch_chunk_size, B)

            # Extract the chunk of batches
            curr_grids_chunk = curr_grids[start:end]  # Shape: (batch_chunk_size, H_map, W_map)
            curr_xy_chunk = curr_xy_stacked[start:end]  # Shape: (batch_chunk_size, H_map, W_map, 2)
            curr_xy_chunk_flat = curr_xy_chunk.reshape(end-start, -1, 2)  # (batch_chunk_size, H_map*W_map, 2)

            # Flatten the grid coordinates for efficient index mapping
            curr_xy_flat = curr_xy_chunk.reshape(end-start, -1, 2)  # (batch_chunk_size, H_map*W_map, 2)
            
            distances, indices = kdtree.query(curr_xy_chunk_flat[0], distance_upper_bound=distance_threshold)  # (batch_chunk_size, H*W)
            mask = distances <= distance_threshold 

            # Using the mask, transfer the values in parallel
            curr_grids_flat = curr_grids_chunk.reshape(end-start, -1) # Flatten curr_grids to (batch_chunk_size, H_map*W_map)

            next_indices = indices[mask]
            # print(curr_xy_chunk.shape, curr_grids.shape, start, end)
            curr_indices = np.arange(H*W)[mask]
            
            updated_grids[start:end].reshape(-1)[next_indices] = curr_grids_flat.reshape(-1)[curr_indices]

        return updated_grids


    def Transfer_to_EgoGrid(self,curr_xy, curr_grids, next_xy, next_grids, res):
        # global x_min, x_max, y_min, y_max
        ###############################################################################################################################
        ## Goal : Transfer pred_maps (in sensor/reference car's grid) cell information to the unknown cells of ego car's sensor_grid
        ## Method : Used global grid as an intermediate (ref indx --> global indx --> ego indx)
        ## return updated_next_grids(N, w', h')
        ## * N : number of agents
        ## * The resolution of global grid should be a little bigger than the local grid's. Else there can be some missing information.
        ###############################################################################################################################
        
        x_min = -6
        x_max = 6
        y_min = -6
        y_max = 6

        global_res = 0.1  #0.2
        global_grid_x, global_grid_y = global_grid(np.array([x_min,y_min]),np.array([x_max,y_max]),global_res)

        x_min = np.min(global_grid_x)
        x_max = np.max(global_grid_x)
        y_min = np.min(global_grid_y)
        y_max = np.max(global_grid_y)
        
        # pred_maps_egoGrid = [] # pred_maps in ego grid

        curr_grids_ = deepcopy(curr_grids)
        # pred_egoGrid = copy.copy(ego_sensor_grid) 
        updated_next_grids = self.transfer_grid_data(curr_xy, curr_grids_, next_xy, next_grids ,distance_threshold=global_res)
        # pred_maps_egoGrid = mask_in_EgoGrid(global_grid_x, global_grid_y, ref_local_xy, ego_xy, pred_egoGrid, pred_maps, global_res)
        # pred_maps_egoGrid.append(pred_egoGrid)

        return updated_next_grids


    def disambig_mask(self, grid_xy, robot_pos, goal, decoded_in_unknown, grid_res):
        """
        Mask the grid to disambiguate the robot's path towards the goal. 
        The mask is a 120 degree cone towards the goal.
        """
        K = len(robot_pos)
        H, W = grid_xy[0].shape
        # kernel_max = 0.5
        sigma = 0.1
        
        disambig_weight_map = np.ones((K, H, W))
        
        # Obtain rays for the robot towards the goal direction with 90 degree FOV
        heading = np.arctan2(goal[1]-robot_pos[:,1], goal[0]-robot_pos[:,0]) # (K,)
        FOV_angle = math.pi/3*2 #np.asin(np.clip(human_radius/np.sqrt((center[:,1]-r_pos[:,1])**2 + (center[:,0]-r_pos[:,0])**2), -1., 1.)) # (K,)
        
        # # Mask points from grid_xy
        x1 = robot_pos[:,0] + 20 * np.cos(heading-FOV_angle/2.) # (K,)
        y1 = robot_pos[:,1] + 20 * np.sin(heading-FOV_angle/2.) # (K,)
        
        x2 = robot_pos[:,0] + 20 * np.cos(heading+FOV_angle/2.) # (K,)
        y2 = robot_pos[:,1] + 20 * np.sin(heading+FOV_angle/2.) # (K,)
        
        polygon = np.stack([np.stack([x1, y1], axis=1), np.stack([x2, y2], axis=1), robot_pos], axis=1) # (K, 2, 2)
        grid_x = grid_xy[[0]].reshape(K, -1)
        grid_y = grid_xy[[1]].reshape(K, -1)
        disambig_mask = self.pointinpolygon(grid_x,grid_y,polygon.transpose(1,0,2)) # (K, H, W)
        disambig_mask = disambig_mask.reshape(K, H, W)
        
        disambig_weight_map[~disambig_mask] = 0.
        
        return disambig_weight_map        



    def compute_disambig_cost(self, robot_pos, grid_xy, sensor_grid, next_sensor_grid, decoded_in_unknown, goal, config):
        """
        For each cell, H(p) = -plogp -(1-p)log(1-p)
        The disambiguation cost is computed on towards the goal direction of the robot with 120 degree cone shaped mask.
        The ambiguation cost is computed on the unobserved area of the sensor grid. 
        The entropy is maximized in the estimated PaS occupancy.
        
        sensor_grid:  [batch_size, H, W] consists of values [0, 0.5, 1] in the same coordinate as decoded_in_unknown.
        decoded_in_unknown: should only be estimating the unobserved areas. Otherwise, it should be the ground truth or zero.        
        """
        grid_res = config.pas.grid_res
        grid_shape = [config.pas.grid_width, config.pas.grid_height] 
        disambig_method = config.reward.disambig_method
        
        disambig_weight_map = self.disambig_mask(grid_xy,robot_pos, goal, decoded_in_unknown, grid_res)

        # Calculate the uncertainty reward
        ## Obtain the unobserved area in both current and next sensor grid and set them to 0.5. Otherwise, 0 for both observed free and occupied cells.
        integrated_sensor = np.zeros(sensor_grid.shape)#.to(decoded_in_unknown.dtype)
        integrated_sensor[np.logical_and(sensor_grid==0.5, next_sensor_grid==0.5)] = 0.5
    
        ## Transfer the PaS estimation to the unknown area of the integrated unknown grid
        # print('integrated', 'decoded',integrated_sensor.shape, decoded_in_unknown.shape)
        integrated_sensor[integrated_sensor==0.5] = decoded_in_unknown[integrated_sensor==0.5] 
        zero_mask = integrated_sensor==0.

        integrated_sensor = integrated_sensor * disambig_weight_map * 0.5 # Making the estimated occupancy the most uncertain.
        disambig_reward_map = -integrated_sensor*np.log(integrated_sensor)-(1-integrated_sensor)*np.log(1-integrated_sensor) #(~zero_mask) # integrated_sensor == 0. or 1. is nan
        disambig_reward_map[np.isnan(disambig_reward_map)] = 0. # make nan to 0.
        disambig_H = np.sum(disambig_reward_map, axis=(1,2))#*coef # (0.03~0.08)*2 np.sum(disambig_weight_map * disambig_reward_map, dim=(1,2))#*coef # (0.03~0.08)*2
        
        return disambig_H, disambig_reward_map, disambig_weight_map



    def step(self, action, decoded=None):
        """
        Compute actions for all agents, detect collision, update environment and return (ob, reward, done, info)
        """
        
        r_pos = np.array([[self.robot.px,self.robot.py]])
        if self.config.reward.disambig_reward_flag and self.config.robot.policy == 'pas_rnn':
            decoded_in_unknown = np.where(self.curr_sensor_grid==0.5, decoded[0,0], 0.0).reshape(1, *self.grid_shape)
            r_goal = np.array([self.robot.gx, self.robot.gy])
            self.H_cur, self.disambig_R_map, self.disambig_W_map = self.compute_disambig_cost(r_pos, self.curr_map_xy, self.curr_sensor_grid, self.curr_sensor_grid, decoded_in_unknown, r_goal, self.config)
            
        if np.all(action==[-99., -99.]) or np.all(action==[-99.]): 
            # Input dummy action values for data collection. To use ORCA planner for robot 
            # Robot's action or state is actually not important for data collection because we only train AE with pedestrian infos.
            with torch.no_grad():
                action = self.robot.act(self.ob)  
        else:
            if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
                # # action=ActionRot(action[0], action[1])
                # [20*(self.config.diffstack.lookahead_steps+1)*4, ]
                # action_raw: (1, 1+human_num, 1+lookahead_steps, 4) 
                action_raw = deepcopy(action.reshape(-1, self.config.diffstack.lookahead_steps+1, 4)) # (1+human_num, 1+lookahead_steps, 4) 
                # action = ActionRot(action_raw[0,0],action_raw[0,1]) # (theta, v)
                if len(action_raw)<2:
                    # repeat action_raw for the candidate
                    action_raw = np.repeat(action_raw, 2, axis=0)
                self.traj_candidates.append(action_raw[1:]) # human trajectories
                self.planned_traj.append(action_raw[0]) # robot trajectory
                          
            else:
                robot_v_prev = np.array([self.robot.vx, self.robot.vy])
                action = self.robot.policy.clip_action(action, self.robot.v_pref, robot_v_prev, self.time_step) # Use previous action to clip
            
                if self.robot.kinematics == 'unicycle':
                    self.desiredVelocity[0] = np.clip(self.desiredVelocity[0]+action.v,-self.robot.v_pref,self.robot.v_pref)
                    self.desiredVelocity[1] = action.r
                    action=ActionRot(self.desiredVelocity[0], self.desiredVelocity[1])          

        human_actions = self.get_human_actions(self.humans, self.robot)

        # # ! currently reward is calculated according to the current human and robot state, not the updated state with current action?
        # # compute reward and episode info
        # reward, done, episode_info = self.calc_reward(action)        
        
        
        # apply action and update all agents
        if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
            # action_raw: (1+human_num, 1+lookahead_steps, 4)
            # manually set the robot position and velocity
            if self.robot.kinematics == 'holonomic':
                r_px, r_py, r_vx, r_vy = action_raw[0][1]
                self.robot.set_diffstack(r_px, r_py, r_vx, r_vy)
                
            else:
                r_px, r_py, r_theta, r_speed = action_raw[0][1]
                # r_vx = r_speed * np.cos(r_theta)
                # r_vy = r_speed * np.sin(r_theta)
                # px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None
                self.robot.set_diffstack(r_px, r_py, r_theta, r_speed)
        else:
            self.robot.step(action)
            
        for i, human_action in enumerate(human_actions):
            self.humans[i].step(human_action)
            
        reward, done, episode_info = self.calc_reward(action)     
            
                
        self.global_time += self.time_step # max episode length=time_limit/time_step

        # compute the observation
        ob = self.generate_ob(reset=False)
        
        if self.config.reward.disambig_reward_flag and self.config.robot.policy == 'pas_rnn':
            next_empty_grid = np.zeros_like(self.curr_sensor_grid)[[-1]]
            ## TODO: make the untransferred areas to be unknown (0.5) in the transferred grid?
            transferred_next_sensor_grid = self.Transfer_to_EgoGrid(ob['grid_xy'],ob['grid'][[-1]], self.curr_map_xy, next_empty_grid, self.grid_res) # decoded in next time (t=1)              
            r_pos = np.array([[self.robot.px,self.robot.py]])
            H_next, disambig_R_map_next, disambig_W_map_next = self.compute_disambig_cost(r_pos, self.curr_map_xy, transferred_next_sensor_grid, self.curr_sensor_grid, decoded_in_unknown, r_goal, self.config)

            disambig_c = (H_next-self.H_cur) * self.config.reward.disambig_factor  # -(-H_next+H_cur) # torch.clamp(H_next-H_cur,min=0.0) 
            
            ## disambig_c added to the reward
            reward += np.any(disambig_c>0) * self.config.reward.disambig_factor  
        

        if self.robot.kinematics == 'unicycle' and self.config.robot.policy != 'pas_mppi':    
            ob['vector'][3:5] =  np.array(action)      # (2, N+1, )   
            
            
        info={'info':episode_info}

        # Update all humans' goals randomly midway through episode
        if self.random_goal_changing:
            if self.global_time % 5 == 0:
                self.update_human_goals_randomly()
            
        # Update a specific human's goal once its reached its original goal
        if self.end_goal_changing:
            for i, human in enumerate(self.humans):
                if norm((human.gx - human.px, human.gy - human.py)) < human.radius:
                    self.update_human_goal(human, i=i)
                    
        if self.config.reward.disambig_reward_flag and self.config.robot.policy == 'pas_rnn':
            self.curr_sensor_grid = deepcopy(ob['grid'][[-1]])
            self.curr_map_xy = deepcopy(np.array(ob['grid_xy'])).reshape(2, *self.grid_shape)
        return ob, reward, done, info    
    
    def lookahead_step(self, robot_states, human_states, lookahead_steps=1.0):
        """
        Update robot and human states for a lookahead step
        """
        lookahead_states = []
        for i in range(lookahead_steps):
            # ! Don't update robot_states. 
            # # TODO: Updating the robot action is not implemented yet, doesn't affect the current implementation.

            # (i) Using ground truth human states/actions                    
            human_actions = self.get_human_actions(human_states, robot_states)
            for i, human_action in enumerate(human_actions):
                human_states[i].step(human_action)
                
            # (ii) TODO: Using the repeated same action of human states 
            
            lookahead_states.append([robot_states.get_diff_state(), *[human.get_diff_state() for human in human_states]])
        return lookahead_states
    


    def render(self, mode='video', all_videoGrids=99., output_file='data/my_model/eval.mp4'):
        """[summary]

        Args:
            mode (str, optional): Haven't updated the 'humna mode'. Use the video mode to plot the FOV grids. Defaults to 'video'.
            all_videoGrids (list, optional): list of FOV grids for the whole episode. Defaults to 99..
            output_file (str, optional): video path/filename  Defaults to 'data/my_model/eval.mp4'.

        Returns:
            [type]: [description]
        """
        import matplotlib.pyplot as plt
        import matplotlib.lines as mlines
        from matplotlib import patches
        if all_videoGrids == 99.:
            pass
        else:
            self.all_videoGrids = all_videoGrids
        plt.rcParams['animation.ffmpeg_path'] = '/usr/bin/ffmpeg'

        x_offset = 0.11
        y_offset = 0.11
        robot_color = '#FFD300' #'yellow'
        goal_color = 'yellow'
        arrow_color = 'red'
        arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        if mode is None: # to pass all_videoGrids to render_traj()
            pass
        elif mode == 'human':
            def calcFOVLineEndPoint(ang, point, extendFactor):
                # choose the extendFactor big enough
                # so that the endPoints of the FOVLine is out of xlim and ylim of the figure
                FOVLineRot = np.array([[np.cos(ang), -np.sin(ang), 0],
                                    [np.sin(ang), np.cos(ang), 0],
                                    [0, 0, 1]])
                point.extend([1])
                # apply rotation matrix
                newPoint = np.matmul(FOVLineRot, np.reshape(point, [3, 1]))
                # increase the distance between the line start point and the end point
                newPoint = [extendFactor * newPoint[0, 0], extendFactor * newPoint[1, 0], 1]
                return newPoint


            ax=self.render_axis
            artists=[]

            # add goal
            goal=mlines.Line2D([self.robot.gx], [self.robot.gy], color=goal_color, marker='*', linestyle='None', markersize=15, label='Goal')
            ax.add_artist(goal)
            artists.append(goal)

            # add robot
            robotX,robotY=self.robot.get_position()

            robot=plt.Circle((robotX,robotY), self.robot.radius, fill=True, color=robot_color)
            ax.add_artist(robot)
            artists.append(robot)

            plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16, loc='upper left')
            
            
            # # Add 120  degree line of field of view for the robot
            # FOV_angle = math.pi/3*2
            # x1 = robotX + self.robot_fov * np.cos(self.robot.theta + FOV_angle/2)
            # y1 = robotY + self.robot_fov * np.sin(self.robot.theta + FOV_angle/2)
            # x2 = robotX + self.robot_fov * np.cos(self.robot.theta - FOV_angle/2)
            # y2 = robotY + self.robot_fov * np.sin(self.robot.theta - FOV_angle/2)
            # FOVLine = mlines.Line2D([robotX, x1], [robotY, y1], linestyle='--')
            # FOVLine2 = mlines.Line2D([robotX, x2], [robotY, y2], linestyle='--')
            # ax.add_artist(FOVLine)
            # ax.add_artist(FOVLine2)
            # artists.append(FOVLine)
            # artists.append(FOVLine2)
            
            # compute orientation in each step and add arrow to show the direction
            radius = self.robot.radius
            arrowStartEnd=[]

            robot_theta = self.robot.theta if self.robot.kinematics == 'unicycle' else np.arctan2(self.robot.vy, self.robot.vx)

            arrowStartEnd.append(((robotX, robotY), (robotX + radius * np.cos(robot_theta), robotY + radius * np.sin(robot_theta))))

            for i, human in enumerate(self.humans):
                theta = np.arctan2(human.vy, human.vx)
                arrowStartEnd.append(((human.px, human.py), (human.px + radius * np.cos(theta), human.py + radius * np.sin(theta))))

            arrows = [patches.FancyArrowPatch(*arrow, color=arrow_color, arrowstyle=arrow_style)
                    for arrow in arrowStartEnd]
            for arrow in arrows:
                ax.add_artist(arrow)
                artists.append(arrow)


            # # draw FOV for the robot
            # # add robot FOV
            # if self.robot_fov < np.pi * 2:
            #     FOVAng = self.robot_fov / 2
            #     FOVLine1 = mlines.Line2D([0, 0], [0, 0], linestyle='--')
            #     FOVLine2 = mlines.Line2D([0, 0], [0, 0], linestyle='--')


            #     startPointX = robotX
            #     startPointY = robotY
            #     endPointX = robotX + radius * np.cos(robot_theta)
            #     endPointY = robotY + radius * np.sin(robot_theta)

            #     # transform the vector back to world frame origin, apply rotation matrix, and get end point of FOVLine
            #     # the start point of the FOVLine is the center of the robot
            #     FOVEndPoint1 = calcFOVLineEndPoint(FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            #     FOVLine1.set_xdata(np.array([startPointX, startPointX + FOVEndPoint1[0]]))
            #     FOVLine1.set_ydata(np.array([startPointY, startPointY + FOVEndPoint1[1]]))
            #     FOVEndPoint2 = calcFOVLineEndPoint(-FOVAng, [endPointX - startPointX, endPointY - startPointY], 20. / self.robot.radius)
            #     FOVLine2.set_xdata(np.array([startPointX, startPointX + FOVEndPoint2[0]]))
            #     FOVLine2.set_ydata(np.array([startPointY, startPointY + FOVEndPoint2[1]]))

            #     ax.add_artist(FOVLine1)
            #     ax.add_artist(FOVLine2)
            #     artists.append(FOVLine1)
            #     artists.append(FOVLine2)
          
            if all_videoGrids == 99.:
                pass
            else:
                self.Con = plt.contourf(self.xy_local_grid[-1][0],self.xy_local_grid[-1][1], self.all_videoGrids[-1][0], cmap='BuPu',alpha=0.8, levels=np.linspace(0, 2, 9)) #, cmap='coolwarm'
                self.cbar = plt.colorbar(self.Con) 
            

            # add humans and change the color of them based on visibility
            human_circles = [plt.Circle(human.get_position(), human.radius, fill=False) for human in self.humans]

            for i in range(len(self.humans)):
                ax.add_artist(human_circles[i])
                artists.append(human_circles[i])

                # green: visible; red: invisible
                if self.detect_visible(self.robot, self.humans[i], robot1=True):
                    human_circles[i].set_color(c='g')
                else:
                    human_circles[i].set_color(c='r')
                plt.text(self.humans[i].px - 0.1, self.humans[i].py - 0.1, str(i), color='black', fontsize=12)

            plt.pause(0.1)
            for item in artists:
                item.remove() # there should be a better way to do this. For example,
                # initially use add_artist and draw_artist later on
            for t in ax.texts:
                t.set_visible(False)
            plt.savefig(output_file+'_'+format(int(self.global_time/self.time_step), '03d')+'.png')
            plt.close()
            
        elif mode == 'video':
            from matplotlib import animation
            import itertools
            self.visible_ids = np.array(self.visible_ids)
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.tick_params(labelsize=16)
            ax.set_xlim(-6, 6)
            ax.set_ylim(-6, 6)
            ax.set_xlabel('x(m)', fontsize=16)
            ax.set_ylabel('y(m)', fontsize=16)
            

            # add robot and its goal
            robot_positions = [state[0].position for state in self.states]
            robot_thetas = [state[0].theta for state in self.states]
            # FOV = plt.Circle(robot_positions[0], 3, fill=False, color='grey',  linestyle='--')
            
            goal = mlines.Line2D([self.robot.gx], [self.robot.gy], color='black', markerfacecolor=robot_color, marker='*', linestyle='None', markersize=15, label='Goal')

            robot = plt.Circle(robot_positions[0], self.robot.radius, fill=True, color=robot_color)
            
            if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
                ## add robot planned trajectory
                # Append the last state to the end of the list to make the length of the list equal to the length of the states list
                self.traj_candidates.append(self.traj_candidates[-1]) # human trajectories
                self.planned_traj.append(self.planned_traj[-1]) # robot trajectory
                
                robot_trajs = self.planned_traj[0][:,:2]
                robot_traj_condidates = self.traj_candidates[0][:,:,:2].reshape(-1,2)
                robot_trajs_candid_c = [plt.Circle(robot_traj_condidates[i], 0.1, fill=False, color='grey') for i in range(len(robot_traj_condidates))]
                for i in range(len(robot_trajs_candid_c)):
                    ax.add_artist(robot_trajs_candid_c[i])
                    
                    
                robot_trajs_c = [plt.Circle(robot_trajs[i], 0.1, fill=False, color=robot_color) for i in range(len(robot_trajs))]
                for i in range(len(robot_trajs_c)):
                    ax.add_artist(robot_trajs_c[i])
                    
                ## add human ground truth trajectory
                humans_lookahead = self.lookahead_list[0]
                human_ahead_c = []
                for t in range(len(humans_lookahead)):
                    human_ahead = [humans_lookahead[t][i][:2]  for i in range(1, self.human_num+1)]
                    human_ahead_c += [plt.Circle(human_ahead[i], 0.1, fill=False, color='grey') for i in range(len(human_ahead))]
                
                for i in range(len(human_ahead_c)):
                    ax.add_artist(human_ahead_c[i])
                        
            ax.add_artist(robot)
            ax.add_artist(goal)
            
            # # Add 120  degree line of field of view for the robot
            # FOV_angle = math.pi/3*2
            # x1 = robot_positions[0][0] + self.robot_fov * np.cos(robot_thetas[0] + FOV_angle/2)
            # y1 = robot_positions[0][1] + self.robot_fov * np.sin(robot_thetas[0] + FOV_angle/2)
            # x2 = robot_positions[0][0] + self.robot_fov * np.cos(robot_thetas[0] - FOV_angle/2)
            # y2 = robot_positions[0][1] + self.robot_fov * np.sin(robot_thetas[0] - FOV_angle/2)
            # FOVLine = mlines.Line2D([robot_positions[0][0], x1], [robot_positions[0][1], y1], linestyle='--')
            # FOVLine2 = mlines.Line2D([robot_positions[0][0], x2], [robot_positions[0][1], y2], linestyle='--')
            # ax.add_artist(FOVLine)
            # ax.add_artist(FOVLine2)
            
            ## Add disambiguation area for the robot towards the goal
            # # Obtain rays for the robot towards the goal direction with 120 degree FOV
            heading = np.arctan2(self.robot.gy-robot_positions[0][1], self.robot.gx-robot_positions[0][0]) # (K,)
            
            # # Mask points from grid_xy
            x1 = robot_positions[0][0] + 20 * np.cos(heading-self.config.robot.disambig_angle*np.pi/2.) # (K,)
            y1 = robot_positions[0][1] + 20 * np.sin(heading-self.config.robot.disambig_angle*np.pi/2.) # (K,)
            
            x2 = robot_positions[0][0] + 20 * np.cos(heading+self.config.robot.disambig_angle*np.pi/2.) # (K,)
            y2 = robot_positions[0][1] + 20 * np.sin(heading+self.config.robot.disambig_angle*np.pi/2.) # (K,)
            
            FOVLine = mlines.Line2D([robot_positions[0][0], x1], [robot_positions[0][1], y1], linestyle='--')
            FOVLine2 = mlines.Line2D([robot_positions[0][0], x2], [robot_positions[0][1], y2], linestyle='--')
            ax.add_artist(FOVLine)
            ax.add_artist(FOVLine2)
    
            
            if self.sim == 'static_obstacles':
                # goal2 = mlines.Line2D([-self.robot.gx], [-self.robot.gy], color='black', markerfacecolor=robot_color, marker='*', linestyle='None', markersize=15, label='Goal2')
                # ax.add_artist(goal2)
                # # ax.add_artist(FOV)
                # plt.legend([robot, goal, goal2], ['Robot', 'Goal', 'Goal2'], fontsize=16, loc='upper left')
                plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16, loc='upper left')
                print('robot goal', self.robot.gx, self.robot.gy)
            else:
                plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16, loc='upper left')

            # add humans and their numbers
            human_positions = [[state[1][j].position for j in range(len(self.humans))] for state in self.states]
            humans = [plt.Circle(human_positions[0][i], self.humans[i].radius, fill=False)
                      for i in range(len(self.humans))]
            human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
            
            for i, human in enumerate(humans):
                ax.add_artist(human)
                ax.add_artist(human_numbers[i])
                if i in list(self.visible_ids[0]):
                    human.set_color(c='blue') # green if seen in the current timestep
                else:
                    human.set_color(c='r') # red if not seen in the current timestep


            # add time annotation
            time = plt.text(-1, 5, 'Time: {}'.format(0), fontsize=16)
            ax.add_artist(time)


            # compute orientation in each step and use arrow to show the direction
            radius = self.robot.radius
            if self.robot.kinematics == 'unicycle':
                # addping np.pi because we consider the robot's orientation as the angle between the x-axis and the direction of the robot
                orientation = [((state[0].px, state[0].py), (state[0].px + radius * np.cos(np.arctan2(state[0].vy, state[0].vx)),
                                                             state[0].py + radius * np.sin(np.arctan2(state[0].vy, state[0].vx)))) for state
                               in self.states]
                orientations = [orientation]
            else:
                orientations = []
                for i in range(self.human_num + 1):
                    orientation = []
                    for state in self.states:
                        if i == 0:
                            agent_state = state[0]
                        else:
                            agent_state = state[1][i - 1]
                        theta = np.arctan2(agent_state.vy, agent_state.vx)
                        orientation.append(((agent_state.px, agent_state.py), (agent_state.px + radius * np.cos(theta),
                                             agent_state.py + radius * np.sin(theta))))
                    orientations.append(orientation)
            arrows = [patches.FancyArrowPatch(*orientation[0], color=arrow_color, arrowstyle=arrow_style)
                      for orientation in orientations]
            for arrow in arrows:
                ax.add_artist(arrow)
            global_step = 0

            plt.savefig(output_file+'_'+format(0, '03d')+'.png')

            def update(frame_num):
                nonlocal global_step
                nonlocal arrows
                nonlocal humans
                nonlocal human_numbers
                nonlocal robot_trajs_c
                nonlocal human_ahead_c
                nonlocal robot_trajs_candid_c
                global_step = frame_num
                
                # # TODO: Add this back
                if self.Con!=None:
                    for coll in self.Con.collections:
                        # ax.collections.remove(coll)
                        coll.remove()
                    # ax.clear()
                if self.cbar!= None:
                    self.cbar.remove()
                    # self.cbar.ax.clear()
                    
                if all_videoGrids == 99.:
                    pass
                else:
                    if np.any(self.all_videoGrids[frame_num] != None):
                        self.Con = plt.contourf(self.xy_local_grid[frame_num][0],self.xy_local_grid[frame_num][1], self.all_videoGrids[frame_num][0], cmap='binary',alpha=0.8, levels=np.linspace(0, 1, 9)) #, cmap='coolwarm' 'BuPu'
                        self.cbar = plt.colorbar(self.Con) 
                        
                        
                robot.center = robot_positions[frame_num]
                
                ## Add disambiguation area for the robot towards the goal
                # # Obtain rays for the robot towards the goal direction with 120 degree FOV
                heading = np.arctan2(self.robot.gy-robot_positions[frame_num][1], self.robot.gx-robot_positions[frame_num][0]) # (K,)
                
                # # Mask points from grid_xy
                x1 = robot_positions[frame_num][0] + 20 * np.cos(heading-self.config.robot.disambig_angle*np.pi/2.) # (K,)
                y1 = robot_positions[frame_num][1] + 20 * np.sin(heading-self.config.robot.disambig_angle*np.pi/2.) # (K,)
                
                x2 = robot_positions[frame_num][0] + 20 * np.cos(heading+self.config.robot.disambig_angle*np.pi/2.) # (K,)
                y2 = robot_positions[frame_num][1] + 20 * np.sin(heading+self.config.robot.disambig_angle*np.pi/2.) # (K,)
                
                
                FOVLine.set_xdata([robot_positions[frame_num][0], x1])
                FOVLine.set_ydata([robot_positions[frame_num][1], y1])
                FOVLine2.set_xdata([robot_positions[frame_num][0], x2])
                FOVLine2.set_ydata([robot_positions[frame_num][1], y2])
                
                if self.config.robot.policy == 'pas_diffstack' or self.config.robot.policy == 'pas_mppi':
                    robot_trajs = self.planned_traj[frame_num][:,:2]
                    robot_trajs_candidates = self.traj_candidates[frame_num][:,:,:2].reshape(-1,2)
                    # FOV.center = robot_positions[frame_num]
                    
                    # update robot trajectory
                    for robot_traj_c in robot_trajs_c:
                        robot_traj_c.remove()
                        
                    for robot_trajs_candid_c in robot_trajs_candid_c:
                        robot_trajs_candid_c.remove()
                        
                    robot_trajs_candid_c = [plt.Circle(robot_trajs_candidates[i], 0.1, fill=False, color='grey') for i in range(len(robot_trajs_candidates))]
                    for i in range(len(robot_trajs_candid_c)):
                        ax.add_artist(robot_trajs_candid_c[i])
                        
                    robot_trajs_c = [plt.Circle(robot_trajs[i], 0.1, fill=False, color=robot_color) for i in range(len(robot_trajs))]
                    for i in range(len(robot_trajs_c)):
                        ax.add_artist(robot_trajs_c[i])
                    
                    ## add human ground truth trajectory
                    for human_ah_c in human_ahead_c:
                        human_ah_c.remove()
                        
                    humans_lookahead = self.lookahead_list[frame_num]
                    human_ahead_c = []
                    for t in range(len(humans_lookahead)):
                        human_ahead = [humans_lookahead[t][i][:2]  for i in range(1, self.human_num+1)]
                        human_ahead_c += [plt.Circle(human_ahead[i], 0.1, fill=False, color='grey') for i in range(len(human_ahead))]
                    
                    for i in range(len(human_ahead_c)):
                        ax.add_artist(human_ahead_c[i])

                for human in humans:
                    human.remove()
                for txt in human_numbers:
                    txt.set_visible(False)
                
                if frame_num >= self.config.pas.sequence:
                    sequence = self.config.pas.sequence
                else:
                    sequence = frame_num+1 # frame_num= 0,1,2,3
                for j in range(sequence):     # frame_num-j >=0 always
                    if j ==0:
                        alpha = 1
                        humans = [plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                for i, human in enumerate(self.states[frame_num][1])]
                        human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
                                      color='black', fontsize=12) for i in range(len(self.humans))]
                    elif j ==1:
                        alpha = 0.4
                        humans.extend([plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                    for i, human in enumerate(self.states[frame_num-1][1])])
                    elif j ==2:
                        alpha = 0.3          
                        humans.extend([plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                    for i, human in enumerate(self.states[frame_num-2][1])])          
                    elif j == 3:
                        alpha = 0.1
                        humans.extend([plt.Circle((human.px, human.py), human.radius, fill=False, color='black', linewidth=1.5, alpha=alpha)
                                    for i, human in enumerate(self.states[frame_num-3][1])])
                    end_frame = frame_num-j                                 
                    
                    
                    for i, human in enumerate(humans[self.human_num*j:self.human_num*(j+1)]):
                        ax.add_artist(human)                
                    
                        # plt.text(self.human_states_copy[frame_num-j][i][0]+0.3, self.human_states_copy[frame_num-j][i][1]+0.3, str((frame_num-j)*self.time_step), fontsize=14, color='black', ha='center', va='center')
                        # Observation history with colored pedestrians
                        if self.config.pas.gridsensor == 'sensor':
                            if self.config.pas.seq_flag or self.config.pas.encoder_type!='cnn':
                                if frame_num-j<self.config.pas.sequence:
                                    start_frame = 0
                                else:
                                    start_frame = frame_num-j - self.config.pas.sequence+1
                                if frame_num-j == 0:
                                    end_frame = 1
                                    
                            
                                past_vis_ids = list(itertools.chain(*list(self.visible_ids[start_frame:end_frame])))
                                if i in list(self.visible_ids[frame_num-j]): # green if seen in current timestep.
                                    human.set_color(c='blue')

                                else:
                                    human.set_color(c='r')
                            else:
                                if i in list(self.visible_ids[frame_num-j]):
                                    human.set_color(c='blue') # green if seen in current timestep.
                                else:
                                    human.set_color(c='r') # red if unseen in current timestep.
                        else: # all agents are observable
                            human.set_color(c='blue') # green if seen in current timestep.

                    for arrow in arrows:
                        arrow.remove()
                    arrows = [patches.FancyArrowPatch(*orientation[frame_num], color=arrow_color,
                                                      arrowstyle=arrow_style) for orientation in orientations]
                    for arrow in arrows:
                        ax.add_artist(arrow)
                 
                time.set_text('Time: {:.2f}'.format(frame_num * self.time_step))
                plt.savefig(output_file+'_'+format(frame_num, '03d')+'.png')
                
            def getint(name):
                basename = name.partition('.')
                phase, epi_num, termination, num = basename.split('_')
                return int(num)
            

            def plot_value_heatmap():
                assert self.robot.kinematics == 'holonomic'
                for agent in [self.states[global_step][0]] + self.states[global_step][1]:
                    print(('{:.4f}, ' * 6 + '{:.4f}').format(agent.px, agent.py, agent.gx, agent.gy,
                                                             agent.vx, agent.vy, agent.theta))
                # when any key is pressed draw the action value plot
                fig, axis = plt.subplots()
                speeds = [0] + self.robot.policy.speeds
                rotations = self.robot.policy.rotations + [np.pi * 2]
                r, th = np.meshgrid(speeds, rotations)
                z = np.array(self.action_values[global_step % len(self.states)][1:])
                z = (z - np.min(z)) / (np.max(z) - np.min(z))
                z = np.reshape(z, (16, 5))
                polar = plt.subplot(projection="polar")
                polar.tick_params(labelsize=16)
                mesh = plt.pcolormesh(th, r, z, vmin=0, vmax=1)
                plt.plot(rotations, r, color='k', ls='none')
                plt.grid()
                cbaxes = fig.add_axes([0.85, 0.1, 0.03, 0.8])
                cbar = plt.colorbar(mesh, cax=cbaxes)
                cbar.ax.tick_params(labelsize=16)
                plt.show()

            def on_click(event):
                anim.running ^= True
                if anim.running:
                    anim.event_source.stop()
                    if hasattr(self.robot.policy, 'action_values'):
                        plot_value_heatmap()
                else:
                    anim.event_source.start()           

            # # Save to mp4
            fig.canvas.mpl_connect('key_press_event', on_click)
            anim = animation.FuncAnimation(fig, update, frames=len(self.states), interval=self.time_step * 1000, repeat=False)
            anim.running = True

            if output_file is not None:
                ffmpeg_writer = animation.writers['ffmpeg']
                writer = ffmpeg_writer(fps=5, metadata=dict(artist='Me'), bitrate=1800)
                anim.save(output_file+'.mp4', writer=writer)
                print(output_file+'.mp4')
            else:
                pass
                # plt.show()
            
            # # # # # # combine saved images to gif
            filenames = glob.glob(output_file+'*.png')
            filenames.sort() # sort by timestep
            images = []
            for filename in filenames:
                images.append(imageio.imread(filename))
                os.remove(filename)
            imageio.mimsave(output_file+'.gif', images, fps=5)
            plt.close()
            
    
    def render_traj(self, path, episode_num=0):
        import matplotlib.pyplot as plt
        from matplotlib import patches
        import matplotlib.lines as mlines
        import itertools
        import os
        
        self.Con = None
        self.cbar = None
        arrows = []
        x_offset = 0.11
        y_offset = 0.11
        # Save current live figure number
        # curr_fig = plt.gcf().number
        # plt.figure(figsize=(7, 7))
        

        # # Set Constants
        # cmap = plt.cm.get_cmap('jet', self.human_num)
        # robot_color = '#FFD300'

        # # Set Axes
        # plt.tick_params(labelsize=16)
        # plt.xlim(-12, 12)
        # plt.ylim(-12, 12)
        # plt.xlabel('x(m)', fontsize=16)
        # plt.ylabel('y(m)', fontsize=16)
        # ax = plt.axes()
        
        
        # # # Begin drawing
        # for k in range(len(self.states)-1): # all_videoGrids does not have the last OGM    
        #     # Save current live figure number
        #     curr_fig = plt.gcf().number
        #     plt.figure(figsize=(7, 7))            

        #     # Set Constants
        #     cmap = plt.cm.get_cmap('jet', self.human_num)
        #     robot_color = '#FFD300'
        #     arrow_color = 'red'
        #     arrow_style = patches.ArrowStyle("->", head_length=4, head_width=2)

        #     # Set Axes
        #     plt.tick_params(labelsize=16)
        #     plt.xlim(-12, 12)
        #     plt.ylim(-12, 12)
        #     plt.xlabel('x(m)', fontsize=16)
        #     plt.ylabel('y(m)', fontsize=16)
        #     ax = plt.axes()
                    
         
        #     robot = plt.Circle((self.states[k][0].px,self.states[k][0].py), self.robot.radius, fill=True, color=robot_color)
        #     ax.add_artist(robot)
        #     theta = np.arctan2(self.states[k][0].vy, self.states[k][0].vx)
        #     orientation = [[((self.states[k][0].px,self.states[k][0].py), (self.states[k][0].px + self.states[k][0].radius * np.cos(theta),
        #                         self.states[k][0].py + self.states[k][0].radius * np.sin(theta)))]]
        
        #     if k >= self.config.pas.sequence:
        #         sequence = self.config.pas.sequence
        #     else:
        #         sequence = k+1 # k= 0,1,2,3
        #     for j in range(sequence):     # k-j >=0 always
        #         if j ==0:
        #             alpha = 1
        #         elif j ==1:
        #             alpha = 0.4
        #         elif j ==2:
        #             alpha = 0.3                    
        #         elif j == 3:
        #             alpha = 0.1
        #         end_frame = k-j                 
        #         humans = [plt.Circle((human.px, human.py), human.radius, fill=False, color=cmap(i), linewidth=1.5, alpha=alpha)
        #                     for i, human in enumerate(self.states[end_frame][1])]
                
        #         for i, human in enumerate(humans):
        #             ax.add_artist(human)                
                
        #             # Observation history with colored pedestrians
        #             if self.config.pas.gridsensor == 'sensor':
        #                 if self.config.pas.seq_flag or self.config.pas.encoder_type!='cnn':
        #                     if k-j<self.config.pas.sequence:
        #                         start_frame = 0
        #                     else:
        #                         start_frame = k-j - self.config.pas.sequence+1
        #                     if k-j == 0:
        #                         end_frame = 1                                
                        
        #                     past_vis_ids = list(itertools.chain(*list(self.visible_ids[start_frame:end_frame])))
        #                     if i in list(self.visible_ids[k-j]): # green if seen in current timestep.
        #                         human.set_color(c='blue')
        #                     elif i in list(np.unique(past_vis_ids)): # red if the not seen in past sequence either.
        #                         human.set_color(c='magenta')
        #                     else:
        #                         human.set_color(c='r')
        #                 else:
        #                     if i in list(self.visible_ids[k-j]):
        #                         human.set_color(c='blue') # green if seen in current timestep.
        #                     else:
        #                         human.set_color(c='r') # red if unseen in current timestep.
        #             else: # all agents are observable
        #                 human.set_color(c='blue') # green if seen in current timestep.
                    

                        
        #         if j == 0:                        
        #             human_numbers = [plt.text(humans[i].center[0] - x_offset, humans[i].center[1] - y_offset, str(i),
        #                                     color='black', fontsize=12) for i in range(len(self.humans))]
        #     for h in range(self.human_num):
        #         ax.add_artist(human_numbers[h])
        #         agent_state = copy.deepcopy(self.states[k][1][h])
        #         theta = np.arctan2(agent_state.vy, agent_state.vx)
        #         orientation.append([((agent_state.px, agent_state.py), (agent_state.px + agent_state.radius * np.cos(theta),
        #                             agent_state.py + agent_state.radius * np.sin(theta)))])

        #     for arrow in arrows:
        #         arrow.remove()
        #     arrows = [patches.FancyArrowPatch(*orient[0], color=arrow_color, arrowstyle=arrow_style)
        #             for orient in orientation]
            
        #     for arrow in arrows:
        #         ax.add_artist(arrow)

                
            # # (i) Drawing for some time intervals
            # # Draw circle for robot and humans every 7 timesteps and at the end of episode
            # # plt.contourf(self.xy_local_grid[k][0],self.xy_local_grid[k][1], self.all_videoGrids[k].cpu().numpy()[0], cmap='binary',alpha=0.8, levels=np.linspace(0, 1, 9)) #, cmap='coolwarm'
            # # plt.colorbar(self.Con)

        
            # # # Draw robot and humans' goals
            # goal = mlines.Line2D([self.states[k][0].gx], [self.states[k][0].gy], color='black', markerfacecolor=robot_color, marker='*', linestyle='None', markersize=15, label='Goal')
            # ax.add_artist(goal)
            # # human_goals = [human.get_goal_position() for human in self.humans]
            # # for i, point in enumerate(human_goals):
            # #     if not self.humans[i].isObstacle:
            # #         curr_goal = mlines.Line2D([point[0]], [point[1]], color='black', markerfacecolor=cmap(i), marker='*', linestyle='None', markersize=15)
            # #         ax.add_artist(curr_goal)

            # # # Draw robot and humans' start positions
            # # for pos in [robot_start_pos] + human_start_poses:
            # #     plt.text(pos[0], pos[1], 'S', fontsize=14, color='black', ha='center', va='center')

            # # Set legend
            # plt.legend([robot, goal], ['Robot', 'Goal'], fontsize=16)

            # # Save trajectory for current episode
            # new_fig = plt.gcf().number
            # plt.savefig(os.path.join(path, str(episode_num) + '_traj'+'_'+str(k*self.time_step*100)+'.png'), dpi=300)

            # # Close trajectory figure and switch back to live figure
            # plt.close(new_fig)
            # plt.figure(curr_fig)  
                
        
        # ## (ii) Drawing the whole episode in one fig
        self.Con = None
        self.cbar = None
        # Save current live figure number
        curr_fig = plt.gcf().number
        # plt.figure(figsize=(7, 7))
        

        # Set Constants
        cmap = plt.cm.get_cmap('jet', self.human_num)
        robot_color = '#FFD300'

        # Set Axes
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.tick_params(labelsize=16)
        ax.set_xlim(-6, 6)
        ax.set_ylim(-6, 6)
        ax.set_xlabel('x(m)', fontsize=16)
        ax.set_ylabel('y(m)', fontsize=16)
        
        
        # # Get robot and humans' start positions
        # robot_start_pos = [self.states[0][0].px, self.states[0][0].py] #self.robot_states_copy[0]
        # human_start_poses = [[human.px, human.py] for human in self.states[0][1]]  #self.human_states_copy[0]       
        
        
        for k in range(len(self.states)-1): # all_videoGrids does not have the last OGM 
            # Draw circle for robot and humans every 7 timesteps and at the end of episode
            if k % 4 == 0 or k == len(self.states) - 1:
                robot = plt.Circle((self.states[k][0].px,self.states[k][0].py), self.robot.radius, fill=True, color=robot_color) 
                # plt.text(self.states[k][0].px+0.3, self.states[k][0].py+0.3, str(k*self.time_step), fontsize=14, color='black', ha='center', va='center')
                humans = [plt.Circle((human.px, human.py), 0.1, fill=False, color=cmap(i), linewidth=1.5) # human.radius
                            for i, human in enumerate(self.states[k][1])]
                ax.add_artist(robot)
                for i, human in enumerate(humans):
                    ax.add_artist(human)
                    
                    if self.config.pas.gridsensor == 'sensor':
                        if self.config.pas.seq_flag or self.config.pas.encoder_type!='cnn':
                            if k<self.config.pas.sequence:
                                start_frame = 0
                            else:
                                start_frame = k - self.config.pas.sequence+1
                            past_vis_ids = list(itertools.chain(*list(self.visible_ids[start_frame:k])))
                            if i in list(self.visible_ids[k]): # green if seen in current timestep.
                                human.set_color(c='blue')
                            elif i in list(np.unique(past_vis_ids)): # red if the not seen in past sequence either.
                                human.set_color(c='magenta')
                            else:
                                human.set_color(c='r')
                        else:
                            if i in list(self.visible_ids[k]):
                                human.set_color(c='blue') # green if seen in current timestep.
                            else:
                                human.set_color(c='r') # red if unseen in current timestep.
                    else: # all agents are observable
                        human.set_color(c='blue') # green if seen in current timestep.        

            # Draw lines for trajectory every step of the episode for all agents
            if k != 0:
                nav_direction = plt.Line2D((self.states[k-1][0].px, self.states[k][0].px),
                                            (self.states[k-1][0].py, self.states[k][0].py),
                                            color=robot_color, ls='solid')
                human_directions = [plt.Line2D((self.states[k-1][1][i].px, self.states[k][1][i].px),
                                                (self.states[k-1][1][i].py, self.states[k][1][i].py),
                                                color=cmap(i), ls='solid')
                                    for i in range(self.human_num)]
                ax.add_artist(nav_direction)
                for human_direction in human_directions:
                    ax.add_artist(human_direction)

        # Draw robot and humans' goals
        goal = mlines.Line2D([self.states[k][0].px], [self.states[k][0].py], color='black', markerfacecolor=robot_color, marker='*', linestyle='None', markersize=15, label='Goal')
        ax.add_artist(goal)
      
        # Set legend
        plt.legend([robot, goal], ['Robot', 'Goal'], loc='upper right', fontsize=16)

        # Save trajectory for current episode
        new_fig = plt.gcf().number
        plt.savefig(os.path.join(path, str(episode_num) + '_traj.png'), dpi=300)

        # Close trajectory figure and switch back to live figure
        plt.close(new_fig)
        plt.figure(curr_fig)
