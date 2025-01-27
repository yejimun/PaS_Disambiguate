"""
<For collecting ae data>
1. Things to change in config
robot.policy = "orca' 
sim.collectingdata = True
limit_path = False

2. Things to change in arguments
'VAEdata/train'

<circle_crossing>
sim.train_val_sim = "circle_crossing"
sim.test_sim = "circle_crossing"
sim.human_num = 7

<escaperoom>
sim.train_val_sim = "escaperoom"
sim.test_sim = "escaperoom"  
sim.human_num = 9
"""


class BaseConfig(object):
    def __init__(self):
        pass
# sim.train_val_sim
# sim.test_sim
# sim.human_num 
# humans.policy
# robot.policy
 

class Config(object):
    env = BaseConfig()
    env.time_limit = 50
    env.time_step = 0.25
    env.val_size = 100
    env.test_size = 500 #500
    env.randomize_attributes = True

    reward = BaseConfig()
    reward.success_reward = 10 # 20 
    reward.collision_penalty = -5 #-20 #-5 #-20 
    reward.timeout_penalty = None #-10
    # discomfort distance for the front half of the robot
    reward.discomfort_dist = 0.25 #0.3 #0.25
    # discomfort distance for the back half of the robot
    reward.discomfort_penalty_factor = 10 #40 #40 #20 
    reward.SumReward = False
    reward.disambig_reward_flag = True # 'True' or 'False'
    reward.disambig_method = 'entropy' # 'linear' or 'entropy'
    reward.disambig_factor = 0.25 #2.0

    sim = BaseConfig()
    sim.collectingdata = False # True # False # ! Set True only when collecting data for AE
    sim.train_val_sim ="circle_crossing"  #"circle_crossing" "crosswalk2" "escaperoom" #  # !
    sim.test_sim = "circle_crossing"  #"circle_crossing" "crosswalk2" "escaperoom" #  # !
    sim.square_width = 10
    sim.circle_radius = 4
    sim.human_num = 12 # ! 9 for escape room / 6 for circle_crossing / 8 crosswalk2
    # Group environment: set to true; FoV environment: false
    sim.group_human = False

    humans = BaseConfig()
    humans.visible = True
    # orca or social_force for now
    humans.policy =  "orca" # "crosswalk"
    humans.radius = 0.3 # ! 0.4 for crosswalk2 / 0.3 for escape room, circle_crossing
    humans.v_pref = 2 # 0.5 #2. #0.65 # !
    humans.sensor = "coordinates"
    # FOV = this values * PI
    humans.FOV = 2.
    humans.limited_path = False # Using 1.5

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = False #True
    humans.goal_change_chance = 0.25

    # a human may change its goal after it reaches its old goal
    humans.end_goal_changing = True
    humans.end_goal_change_chance = 1.0

    # a human may change its radius and/or v_pref after it reaches its current goal
    humans.random_radii = False
    humans.random_v_pref = False

    # one human may have a random chance to be blind to other agents at every time step
    humans.random_unobservability = False
    humans.unobservable_chance = 0.3

    humans.random_policy_changing = False

    robot = BaseConfig()
    robot.visible = False 
    # srnn for now
    robot.policy =  'pas_rnn'  #'orca'   # ! 
    robot.radius = 0.3
    robot.v_pref = 2 # 0.5 #2. #0.65 # ! 
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2.
    robot.FOV_radius = 3.0
    robot.limited_path = False  
    robot.onedim_action = False 
    robot.loop = 1

    noise = BaseConfig()
    noise.add_noise = False
    # uniform, gaussian
    noise.type = "uniform"
    noise.magnitude = 0.1

    action_space = BaseConfig()
    # holonomic or unicycle
    action_space.kinematics = "holonomic" # !

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.15
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    
    # config for pas_rnn
    pas = BaseConfig()
    pas.grid_res = 0.1
    pas.grid_width = 100
    pas.grid_height = 100
    pas.gridsensor = 'sensor' #'sensor' # 'gt' # !
    pas.gridtype = 'local' # 'local' 'global'
    pas.sequence = 4 # number of FOV grids stacked for Sensor AE lstm, past + present
    pas.encoder_type = 'vae'  #'vae' or 'cnn'
    pas.PaS_coef = 0. 
    pas.seq_flag = True
    
    # config for diffstack
    diffstack = BaseConfig()
    diffstack.lookahead_steps = 4
    
    # mppi
    diffstack.lambda_ = 1.
    diffstack.num_samples = 100


    
    # pas.occ_prob = 0.6 # for  # !
    # pas.overest = False # For reconstruction loss. If True, impose more penalty for estimating empty on actually occupied cell.
    # pas.sequence = 4 # number of FOV grids stacked for Sensor AE lstm # !
    # pas.encoder_type = 'vae'  #'autoencoder' #'vae'  'cnn'
    # pas.gridwego = False #True # True if the robot needs to be included in the grid
    # # pas.fuse_flag = False # For training Label/Sensor AE. True if reconstructing only the occluded pedestrians. False for reconstructing occluded & visible humans
    # pas.is_recurrent = True
    # pas.m_coef = 1. # 1. # -1. # 10 # !
    # pas.est_coef = 1.
    # pas.m_decoder = False
    # pas.seq_flag = True # !


# raw_action[0] = np.clip(raw_action[0], -0.65/self.time_step, 0.65/self.time_step) # action[0] is change of v # Max is 1.0?
# raw_action[1] = np.clip(raw_action[1], -2./self.time_step, 2./self.time_step) # action[1] is change of theta # Max is 2.0 or 3.14?