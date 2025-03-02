import numpy as np

class BaseConfig(object):
    def __init__(self):
        pass

class Config(object):
    env = BaseConfig()
    env.time_limit = 50 
    env.time_step = 0.25 #0.25
    env.val_size = 100
    env.test_size = 500  # TODO: change back to 500
    env.randomize_attributes = False # False for turtlebot experiment

    reward = BaseConfig()
    reward.success_reward = -5 
    reward.collision_penalty = 100
    reward.timeout_penalty = None 
    reward.discomfort_dist = 0.4 # 0.3
    reward.discomfort_penalty_factor = 3 #0.3 #4 # 2 #10 
    reward.disambig_reward_flag = True # 'True' or 'False'
    reward.disambig_method = 'entropy' # 'linear' or 'entropy'
    reward.disambig_factor = 0.3 #0.1 #0.003 # 10.#0.01 # 0.003

    sim = BaseConfig()
    sim.collectingdata = False #False #False # or True  
    sim.train_val_sim = "circle_crossing" #'static_obstacles' # "circle_crossing" #"circle_crossing"  # 'static_obstacles'
    sim.test_sim = "circle_crossing" #'static_obstacles' # "circle_crossing" #"circle_crossing"  # 'static_obstacles'
    sim.square_width = 10
    sim.circle_radius = 5
    sim.human_num = 20 # 12 #6 # 12  # 6 # 4 for turtlebot experiment
    # 'truth': ground truth future traj
    # 'const_vel': constant velocity model,
    # 'inferred': inferred future traj from prediction model
    sim.predict_method = 'inferred' # using "CollisionGrid" model

    humans = BaseConfig()
    humans.visible = True
    humans.policy =  "orca" #"social_force" # "orca"
    humans.radius = 0.3 #0.5 #0.3  # TODO: change back to 0.3
    humans.v_pref = 2. # 0.5 for the turtlebot experiment
    humans.sensor = "coordinates"
    # FOV = this values * PI
    humans.FOV = 2.

    # a human may change its goal before it reaches its old goal
    humans.random_goal_changing = False 
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
    robot.policy = 'pas_mppi' #'pas_mppi' #'pas_diffstack' #'pas_rnn'  #'orca' 
    robot.radius = 0.3
    robot.w_max = 0.5 # for unicycle. robot.w_max*PI
    robot.v_pref = 2 #2 # 0.5 for the turtlebot experiment
    robot.sensor = "coordinates"
    # FOV = this values * PI
    robot.FOV = 2. # radius of the FOV
    robot.FOV_radius = 2.5
    robot.disambig_angle = 0.6
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
    action_space.kinematics = "unicycle" #"holonomic"  # unicycle for the turtlebot experiment

    # config for ORCA
    orca = BaseConfig()
    orca.neighbor_dist = 10
    orca.safety_space = 0.6 # 0.25 for the turtlebot experiment
    orca.time_horizon = 5
    orca.time_horizon_obst = 5

    # config for social force
    sf = BaseConfig()
    sf.A = 1
    sf.B = 1
    sf.KI = 1
    
    # config for pas_rnn
    pas = BaseConfig()
    pas.grid_res = 0.1
    pas.grid_width = 100
    pas.grid_height = 100
    pas.gridsensor = 'sensor' #'sensor' or 'gt' 
    pas.gridtype = 'local' 
    pas.sequence = 4 # number of FOV grids stacked for Sensor AE lstm, past + present
    pas.encoder_type = 'vae'  #'vae' or 'cnn'
    pas.PaS_coef = 1. 
    pas.est_coef = 0.
    pas.seq_flag = True
    
    # config for diffstack
    diffstack = BaseConfig()
    diffstack.lookahead_steps = 4
    
    # mppi
    diffstack.lambda_ = 0.1
    diffstack.num_samples = 50

