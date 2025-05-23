import numpy as np
from numpy.linalg import norm
import abc
import logging
from crowd_nav.policy.policy_factory import policy_factory
from crowd_sim.envs.utils.action import ActionXY, ActionRot
from crowd_sim.envs.utils.state import ObservableState, FullState, ObservableState_noV
import pdb

class Agent(object):
    def __init__(self, config, section, policy=None):
        """
        Base class for robot and human. Have the physical attributes of an agent.
        """
        subconfig = config.robot if section == 'robot' else config.humans
        self.visible = subconfig.visible
        self.v_pref = subconfig.v_pref
        self.radius = subconfig.radius
        if policy is None:
            self.policy = policy_factory[subconfig.policy](config)
        elif policy == 'none':
            self.policy = policy_factory[policy]
        else:
            self.policy = policy_factory[policy](config)
        self.sensor = subconfig.sensor
        self.FOV = np.pi * subconfig.FOV
        # for humans: we only have holonomic kinematics; for robot: depend on config
        self.kinematics = 'holonomic' if section == 'humans' else config.action_space.kinematics
        self.px = None
        self.py = None
        self.gx = None
        self.gy = None
        self.vx = None
        self.vy = None
        self.ax = None
        self.ay = None
        self.theta = None
        self.time_step = config.env.time_step
        self.policy.time_step = config.env.time_step


    def print_info(self):
        logging.info('Agent is {} and has {} kinematic constraint'.format(
            'visible' if self.visible else 'invisible', self.kinematics))


    def sample_random_attributes(self):
        """
        Sample agent radius and v_pref attribute from certain distribution
        :return:
        """
        self.v_pref = np.random.uniform(0.5, 1.5)
        self.radius = np.random.uniform(0.3, 0.4)

    def set(self, px, py, gx, gy, vx, vy, theta, radius=None, v_pref=None):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.v = np.sqrt(vx**2 + vy**2)
        self.theta = theta

        if radius is not None:
            self.radius = radius
        if v_pref is not None:
            self.v_pref = v_pref
            
    def set_diffstack(self, px, py, a1, a2):
        self.px = px
        self.py = py
        if self.kinematics == 'holonomic': # MPPI always holonomic for now
            self.vx = a1 # actually vx
            self.vy = a2 # actually vy  
            self.v = np.sqrt(self.vx**2 + self.vy**2)
            self.theta = np.arctan2(self.vy, self.vx)     
        else:
            self.v = a1
            self.theta = (self.theta + a2) % (2 * np.pi) #- np.pi
            self.vx = self.v * np.cos(self.theta)
            self.vy = self.v * np.sin(self.theta)
        
    # self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta
    def set_list(self, px, py, vx, vy, radius, gx, gy, v_pref, theta):
        self.px = px
        self.py = py
        self.gx = gx
        self.gy = gy
        self.vx = vx
        self.vy = vy
        self.theta = theta
        self.radius = radius
        self.v_pref = v_pref

    def get_observable_state(self):
        return ObservableState(self.px, self.py, self.vx, self.vy, self.radius, self.visible)

    def get_observable_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius]

    def get_observable_state_noV(self):
        return ObservableState_noV(self.px, self.py, self.radius)

    def get_observable_state_list_noV(self):
        return [self.px, self.py, self.radius]

    def get_next_observable_state(self, action):
        self.check_validity(action)
        pos = self.compute_position(action, self.time_step)
        next_px, next_py = pos
        if self.kinematics == 'holonomic':
            next_vx = action.vx
            next_vy = action.vy                
        else:
            next_theta = self.theta + action.r
            next_vx = action.v * np.cos(next_theta)
            next_vy = action.v * np.sin(next_theta)

        return ObservableState(next_px, next_py, next_vx, next_vy, self.radius, self.visible)

    def get_full_state(self):
        return FullState(self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta, self.visible)

    def get_full_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.v_pref, self.theta]

    def get_full_state_list_noV(self):
        return [self.px, self.py, self.radius, self.gx, self.gy, self.v_pref, self.theta]

    def get_key_state_list_noV(self):
        return [self.px, self.py, self.radius, self.gx, self.gy, self.theta]
    
    def get_key_state_list(self):
        return [self.px, self.py, self.vx, self.vy, self.radius, self.gx, self.gy, self.theta]
    
    def get_diff_state(self):
        # self.v = np.sqrt(self.vx**2 + self.vy**2)     
        # self.theta = np.arctan2(self.vy, self.vx)
        if self.kinematics == 'holonomic':
            return [self.px, self.py, self.vx, self.vy, self.gx, self.gy, self.radius]
        else:
            return [self.px, self.py, self.theta, self.v, self.gx, self.gy, self.radius]

    def get_position(self):
        return self.px, self.py

    def set_position(self, position):
        self.px = position[0]
        self.py = position[1]

    def get_goal_position(self):
        return self.gx, self.gy

    def get_velocity(self):
        return self.vx, self.vy


    def set_velocity(self, velocity):
        self.vx = velocity[0]
        self.vy = velocity[1]


    @abc.abstractmethod
    def act(self, ob):
        """
        Compute state using received observation and pass it to policy
        """
        return

    def check_validity(self, action):
        if self.kinematics == 'holonomic':
            assert isinstance(action, ActionXY)
        else:
            assert isinstance(action, ActionRot)

    def compute_position(self, action, delta_t):
        self.check_validity(action)
        if self.kinematics == 'holonomic':
            px = self.px + action.vx * delta_t
            py = self.py + action.vy * delta_t

            
        # unicycle
        else:
            # differential drive
            epsilon = 0.0001
            if abs(action.r) < epsilon:
                #### TODO: For turtlebot experiment I think. Pure rotation. Then, uncomment the below two lines for px, py.
                # R = 0 
                # px = self.px - R * np.sin(self.theta) + R * np.sin(self.theta + action.r)
                # py = self.py + R * np.cos(self.theta) - R * np.cos(self.theta + action.r)
                ##########################
                px = self.px + action.v * delta_t * np.cos(self.theta)
                py = self.py + action.v * delta_t * np.sin(self.theta)
            else:
                w = action.r/delta_t # action.r is delta theta
                R = action.v/w

                px = self.px - R * np.sin(self.theta) + R * np.sin(self.theta + action.r)
                py = self.py + R * np.cos(self.theta) - R * np.cos(self.theta + action.r)


        return px, py

    def step(self, action):
        """
        Perform an action and update the state
        """
        self.check_validity(action)
        last_x, last_y = self.px, self.py
        last_vx, last_vy = self.vx, self.vy
        pos = self.compute_position(action, self.time_step)
        self.px, self.py = pos
        
        if self.kinematics == 'holonomic':
            self.vx = action.vx
            self.vy = action.vy            
            self.theta = np.arctan2(self.vy, self.vx)
        else:
            self.theta = (self.theta + action.r) % (2 * np.pi) # action.r: angular change, not angular velocity (w).
            # Using updated theta just for visualization purpose. 
            self.vx = action.v * np.cos(self.theta)
            self.vy = action.v * np.sin(self.theta)

        self.ax = (self.vx-last_vx) / self.time_step
        self.ay = (self.vy-last_vy) / self.time_step

    def reached_destination(self):
        return norm(np.array(self.get_position()) - np.array(self.get_goal_position())) < self.radius