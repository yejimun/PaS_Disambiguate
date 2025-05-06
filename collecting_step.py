import numpy as np
import torch
from crowd_sim.envs.utils.info import *
from mppi.mppi import MPPI_Planner
import os
import pdb


def CollectingStep(args, config, model_dir, data_envs, device, test_size, logging, visualize=False):
    data_total_timesteps = 0
    baseEnv = data_envs.venv.envs[0].env
    device = torch.device("cuda" if args.cuda else "cpu")
    video_dir = model_dir+"/data_video/"
    
    if config.robot.policy == 'pas_mppi':
        mppi_module = MPPI_Planner(config, args, device)
        sensor_grid_key = 'grid'
        vector_key = 'mppi_vector'
        sequence = config.pas.sequence
    else:
        sensor_grid_key = 'sensor_grid'
        vector_key = 'vector'

    for k in range(test_size):
        done = False
        all_videoGrids = []
        stepCounter = 0
        obs = data_envs.reset()
        
        ## updated 5/5
        if np.random.random() < 0.5:
            mppi_module.disambig_reward_flag = True
        else:
            mppi_module.disambig_reward_flag = False
        # print(k, mppi_module.disambig_reward_flag)
        #############
        
        masks = torch.FloatTensor([[1.0]]).to(device)
        global_time = 0.0
        total_timesteps = int(config.env.time_limit/config.env.time_step)
        
        ego_data = [['episode', 'timestamp', 'obs', 'label_grid', 'sensor_grid', 'id_grid', 'mask']]
        
        # # 'orca' : 'vector' [1, 1, 59], 'label_grid' [1, 2, 100, 100], 'sensor_grid' [1, 1, 100, 100]
        # # 'pas_mppi' : 'vector' [1, 1, 59], 'label_grid' [1, 2, 100, 100], 'grid' [1, 2, 100, 100]
        # print(obs['vector'].size(), obs['label_grid'].size(), obs['grid'].size(), obs['mppi_vector'].size())
        # pdb.set_trace()
        
        if config.robot.policy == 'pas_mppi':
            ego_step_data = [k, global_time, obs['vector'].cpu().numpy(), obs['label_grid'][:,[0]].cpu().numpy(), obs['grid'][:,[-1]].cpu().numpy(),\
                obs['label_grid'][:,[1]].cpu().numpy(), masks.cpu().numpy()] 
        else:
            ego_step_data = [k, global_time, obs['vector'].cpu().numpy(), obs['label_grid'][:,[0]].cpu().numpy(), obs['sensor_grid'].cpu().numpy(),
                obs['label_grid'][:,[1]].cpu().numpy(), masks.cpu().numpy()] 
        ego_data.append(ego_step_data)
        data_total_timesteps += 1

        while not done:
            with torch.no_grad():
                stepCounter = stepCounter + 1 
                if stepCounter==total_timesteps:
                    break
                    
                if visualize and k<15:      
                    # print(obs['sensor_grid'].cpu().numpy())
                    if config.robot.policy == 'pas_mppi':
                        all_videoGrids.append(obs['grid'][0,[-1]].cpu().numpy())
                    else:
                        all_videoGrids.append(obs['sensor_grid'][0].cpu().numpy())
                else:
                    all_videoGrids = torch.Tensor([99.])
                
                if config.robot.policy == 'orca':
                    action = torch.Tensor([-99., -99.])
                elif config.robot.policy == 'pas_mppi':
                    decoded = obs['label_grid'][:,0].clone()
                    # ignore the walls
                    decoded = torch.where(obs['label_grid'][:,1]==-9999., torch.tensor(0.0), decoded)
                    action, _ = mppi_module.plan(obs, decoded, baseEnv.wall_polygons, config) # decoded is not actually used since we don't use disambiguation.

                # Obser reward and next obs
                obs, rew, done, infos = data_envs.step(action)
                
                if not done:
                    global_time = baseEnv.global_time
    
                masks = torch.FloatTensor(
                    [[0.0] if done_ else [1.0] for done_ in done]).to(device)
                
                if done and visualize and k<15:      
                    # print(obs['sensor_grid'].cpu().numpy())
                    if config.robot.policy == 'pas_mppi':
                        all_videoGrids.append(obs['grid'][0,[-1]].cpu().numpy())
                    else:
                        all_videoGrids.append(obs['sensor_grid'][0].cpu().numpy())
                    
                    if not os.path.exists(video_dir):
                        os.makedirs(video_dir) 
                    data_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=video_dir+'data_epi_'+str(k))
                        
                if config.robot.policy == 'pas_mppi':
                    ego_step_data = [k, global_time, obs['vector'].cpu().numpy(), obs['label_grid'][:,[0]].cpu().numpy(), obs['grid'][:,[-1]].cpu().numpy(),\
                        obs['label_grid'][:,[1]].cpu().numpy(), masks.cpu().numpy()] 
                else:
                    ego_step_data = [k, global_time, obs['vector'].cpu().numpy(), obs['label_grid'][:,[0]].cpu().numpy(), obs['sensor_grid'].cpu().numpy(),\
                        obs['label_grid'][:,[1]].cpu().numpy(), masks.cpu().numpy()] 
            
                # ego_step_data = [k, global_time, obs['vector'].cpu().numpy(), obs['label_grid'][:,[0]].cpu().numpy(), obs['sensor_grid'].cpu().numpy(), \
                #     obs['label_grid'][:,[1]].cpu().numpy(), masks.cpu().numpy()]
                ego_data.append(ego_step_data)
                data_total_timesteps += 1

  
        # if visualize and k<15:
        #     if not os.path.exists(video_dir):
        #         os.makedirs(video_dir) 
        #     data_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=video_dir+'data_epi_'+str(k))
                
            
        ego_data = np.array(ego_data, dtype=object)
        np.save(model_dir+'/epi_'+format(k, '05d')+'_vector', np.array(np.vstack(ego_data[1:,2]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_label_grid', np.array(np.vstack(ego_data[1:,3]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_sensor_grid', np.array(np.vstack(ego_data[1:,4]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_id_grid', np.array(np.vstack(ego_data[1:,5]), dtype=np.float32))
        np.save(model_dir+'/epi_'+format(k, '05d')+'_mask', np.array(np.vstack(ego_data[1:,-1]), dtype=np.float32))

        print('Episode', k, 'ends in', stepCounter)    
    
    logging.info('Total data timesteps: (%d,) ',data_total_timesteps)            
    data_envs.close()