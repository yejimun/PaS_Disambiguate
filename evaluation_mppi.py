import numpy as np
import torch
import os
import pdb
from typing import Dict
from collections import defaultdict
import time
from crowd_sim.envs.utils.info import *
from crowd_sim.envs.grid_utils import MapSimilarityMetric


def average(input_list):
    if input_list:
        return sum(input_list) / len(input_list)
    else:
        return 0
    
def plot_ego_states(dt, trajectories, save_dir, k):
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    from matplotlib.patches import Circle
    
    save_dir =  os.path.join(save_dir, 'val_render')

    # plot robot's x, y, theta, v in each plot
    fig, axes = plt.subplots(len(trajectories[0]))
    fig.suptitle('Robot trajectory')
    fig.tight_layout()
    labels = ['x', 'y','theta','v']
    trajectories = np.stack(trajectories)
    for i, data in enumerate(zip(labels, trajectories.T)):
        label, trajectory = data
        x_linspace = np.linspace(0,len(trajectory)*dt, len(trajectory))
        axes[i].plot(x_linspace, trajectory)
        axes[i].set_title(label)
        
    os.makedirs(save_dir) if not os.path.exists(save_dir) else None
    save_path = os.path.join(save_dir, f'ego_states_epi_{k}.png')
    plt.savefig(save_path)
    plt.close()
    
    
    

def evaluate(rollouts, config, model_dir, label_vae, sensor_vae, mppi_module, eval_envs, device, test_size, logging, visualize=False,
             phase=None, j=None):

    eval_episode_rewards = []
    
    if sensor_vae is not None:
        sensor_vae.eval()

    success_times = []
    collision_times = []
    timeout_times = []
    path_lengths = []
    chc_total = []
    success = 0
    collision = 0
    timeout = 0
    too_close = 0.
    min_dist = []
    cumulative_rewards = []
    collision_cases = []
    timeout_cases = []
    gamma = 0.99
    baseEnv = eval_envs.venv.envs[0].env    
    
    
    # total_similarity = []
    # total_occupied_similarity = []
    # total_free_similarity = []
    # total_occluded_similarity = []
    # total_base_similarity = []
    # total_base_occupied_similarity = []
    # total_base_free_similarity = []
    # total_base_occluded_similarity = []
    
    # epoch_metrics: Dict[str, list] = defaultdict(list)
    

    for k in range(test_size):
        done = False
        rewards = []
        stepCounter = 0
        episode_rew = 0
        all_videoGrids = []
        trajectories = []
        print('Episode', k, '/ Disambig_flag:', config.reward.disambig_reward_flag)
        obs = eval_envs.reset()
        
        if rollouts is not None:
            rollouts.reset()
            if isinstance(obs, dict):
                for key in obs:
                    rollouts.obs[key][0].copy_(obs[key])
            else:
                rollouts.obs[0].copy_(obs)

            rollouts.to(device)

            
            eval_recurrent_hidden_states = {}
            for key in rollouts.recurrent_hidden_states:
                eval_recurrent_hidden_states[key] = rollouts.recurrent_hidden_states[key][stepCounter]
        
        global_time = 0.0
        path = 0.0
        chc = 0.0
        start_time = 0.0

    
        last_pos = obs['vector'][0, 0, :2].cpu().numpy()  # robot px, py
        if config.action_space.kinematics == 'unicycle':
            last_angle = obs['vector'][0, 0, 2].cpu().numpy() 
                     
        while not done:         
            with torch.no_grad():
                if rollouts is not None:
                    masks = rollouts.masks[stepCounter]               
               
                # _, _, z= sensor_vae(obs['grid'])
                # decoded = label_vae.decoder(z) # (1,1,100,100)
                                                
                ### Using the ground truth occupancy of all human agents.
                decoded = obs['label_grid'][:,0].clone()
                # ignore the walls
                decoded = torch.where(obs['label_grid'][:,1]==-9999., torch.tensor(0.0), decoded)
                # # only care about the agents that are fully occluded
                # human_ids = torch.unique(obs['label_grid'][:,[1]])
                # human_ids = human_ids[torch.logical_and(human_ids!=-9999., ~torch.isnan(human_ids))]
                # for h_id in human_ids:
                #     h_indices = torch.where(obs['label_grid'][:,[1]]==h_id)
                #     if torch.any(obs['grid'][[[-1]]][h_indices]==1.): # if human is partially seen, remove from the decoded
                #         decoded[h_indices]=0.
                        
                    
                
                # To test the disambiguation, manually place a estimation in the decoded grid in the unknown area. 30 degree from the robot's heading
                # decoded[0,0,80:90,80:90] = 1.0
                
                action, disambig_R_map = mppi_module.plan(obs, decoded, baseEnv.wall_polygons, config)
                # Use the observation grid for disambiguation. Set only the unknown area to be 1.
                # obs_unknown_mask = torch.where(obs['grid'][:,-1]==0.5, torch.tensor(1.0), torch.tensor(0.0))
                # action, disambig_R_map = mppi_module.plan(obs, obs_unknown_mask, config)
                # else: # if robot's policy is ORCA 
                #     action = torch.Tensor([-99., -99.])


            if not done:
                global_time = baseEnv.global_time
                
            if visualize: # and (k==59 or k==85):
                # all_videoGrids.append(obs['label_grid'][:,0].cpu().numpy())
                all_videoGrids.append(obs['grid'][:,-1].cpu().numpy())
                trajectories.append(obs['mppi_vector'][0, config.pas.sequence-1, 0, :4].cpu().numpy())
                # decoded_only_unknown = torch.where(obs['grid'][:,-1]==0.5, decoded[:,0], torch.tensor(0.0))
                # all_videoGrids.append(decoded_only_unknown.cpu().numpy())
                # all_videoGrids.append(disambig_R_map.cpu().numpy()[[0]])
                # all_videoGrids.append(obs['grid'][:,-1].cpu().numpy())
                # if config.robot.policy == 'pas_rnn':
                #     # if config.pas.encoder_type == 'vae' and config.pas.gridsensor == 'sensor':
                #     #     all_videoGrids.append(decoded.squeeze(0).squeeze(1).cpu().numpy())
                #     # else:
                #     all_videoGrids.append(obs['grid'][:,-1].cpu().numpy())
                        
                # else:
                #     all_videoGrids = torch.Tensor([99.])

            obs, rew, done, infos = eval_envs.step(action)
            
            # pdb.set_trace()
            # print(time.time()-start_time) # takes 1.8~1.9s for each step with disambiguation
            # start_time = time.time()   

            path = path + np.linalg.norm(np.array([last_pos[0] - obs['vector'][0, 0, 0].cpu().numpy(),last_pos[1] - obs['vector'][0, 0, 1].cpu().numpy()]))


            if config.action_space.kinematics == 'unicycle':
                chc = chc + abs(obs['vector'][0, 0, 2].cpu().numpy() - last_angle)
            last_angle = obs['vector'][0, 0, 2].cpu().numpy() 

            last_pos = obs['vector'][0, 0, :2].cpu().numpy()  
            

            rewards.append(rew)


            if isinstance(infos[0]['info'], Danger):
                too_close = too_close + 1
                min_dist.append(infos[0]['info'].min_dist)

            episode_rew += rew[0]

            # If done then clean the history of observations.
            mask = torch.Tensor(
				[[0.0] if done_ else [1.0] for done_ in done])#.cuda()


            if config.robot.policy=='pas_rnn' and config.pas.encoder_type != 'cnn':
                masks = torch.cat([masks[ :, 1:], mask],-1)
            else:
                masks = mask

            if rollouts is not None:
                if config.robot.policy == 'pas_mppi': # Not used
                    action_ = action.reshape(-1, config.diffstack.lookahead_steps+1, 4)[0].reshape(1,-1)
                    rollouts.insert(obs, eval_recurrent_hidden_states, action_,
                                    rewards=rew, masks=masks)
            else:
                    rollouts.insert(obs, eval_recurrent_hidden_states, action,
                                    rewards=rew, masks=masks)

            for info in infos:
                if 'episode' in info.keys():
                    eval_episode_rewards.append(info['episode']['r'])                    
            
            if done and visualize and (k <10 or isinstance(infos[0]['info'], Collision)): # and (k==59 or k==85):        
            # if done and visualize and  isinstance(infos[0]['info'], Collision): # and (k==59 or k==85):  
                trajectories.append(obs['mppi_vector'][0, config.pas.sequence-1, 0, :4].cpu().numpy())  # robot px, py, theta, v
                plot_ego_states(config.env.time_step, trajectories, model_dir, k)
                
                ### Using the ground truth occupancy of all human agents.
                decoded = obs['label_grid'][:,0].clone()
                # ignore the walls
                decoded = torch.where(obs['label_grid'][:,1]==-9999., torch.tensor(0.0), decoded)                
                      
                action, disambig_R_map = mppi_module.plan(obs, decoded, baseEnv.wall_polygons, config)
                # all_videoGrids.append(obs['label_grid'][:,0].cpu().numpy())
                all_videoGrids.append(obs['grid'][:,-1].cpu().numpy())
                # decoded_only_unknown = torch.where(obs['grid'][:,-1]==0.5, decoded[:,0], torch.tensor(0.0))
                # all_videoGrids.append(disambig_R_map.cpu().numpy()[[0]])
                # all_videoGrids.append(decoded_only_unknown.cpu().numpy())
                
                video_dir = model_dir+"/"+phase+"_render/"
                if not os.path.exists(video_dir):
                    os.makedirs(video_dir) 
                    
                output_file=video_dir+str(j)+'_'+'eval_epi'+str(k)+'_'+str(infos[0]['info'])
                if phase == 'val':
                    eval_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=output_file) 
                else:
                    eval_envs.render(mode='video', all_videoGrids=all_videoGrids, output_file=output_file)  # mode='video'  or  mode=None for render_traj          
                
            stepCounter = stepCounter + 1    
            # print("step", stepCounter)        
           
        if phase=='test':
            print('')
            print('Reward={}'.format(episode_rew))              
            print('Episode', k, 'ends in', stepCounter+1)
        

        if isinstance(infos[0]['info'], ReachGoal):
            success += 1
            success_times.append(global_time)
            if phase=='test':
                print('Success')
            path_lengths.append(path)
            if config.action_space.kinematics == 'unicycle':
                chc_total.append(chc)
            
        elif isinstance(infos[0]['info'], Collision):
            collision += 1
            collision_cases.append(k)
            collision_times.append(global_time)
            if phase=='test':
                print('Collision')
        elif isinstance(infos[0]['info'], Timeout):  
            timeout += 1
            timeout_cases.append(k)
            timeout_times.append(baseEnv.time_limit)
            if phase=='test':
                print('Time out')
        else:
            print(infos[0]['info'])
            raise ValueError('Invalid end signal from environment')

        cumulative_rewards.append(sum([pow(gamma, t * baseEnv.robot.time_step * baseEnv.robot.v_pref)
                                       * reward for t, reward in enumerate(rewards)]).item())
    success_rate = success / test_size
    collision_rate = collision / test_size
    timeout_rate = timeout / test_size
    assert success + collision + timeout == test_size

    extra_info = ''
    logging.info(
        '{:<5} {}has success rate: {:.2f}, collision rate: {:.2f}, timeout rate: {:.2f}, '
        'nav time (mean/var): {:.2f}/{:.2f}, total reward: {:.4f}'.
            format(phase.upper(), extra_info, success_rate, collision_rate, timeout_rate, np.mean(success_times), np.var(success_times),
                np.average((cumulative_rewards))))
    if phase in ['val', 'test']:
        total_time = sum(success_times + collision_times + timeout_times)
        if min_dist:
            avg_min_dist = np.average(min_dist)
        else:
            avg_min_dist = float("nan")
        logging.info('Frequency of being in danger: %.2f and average min separate distance in danger: %.2f',
                    too_close * baseEnv.robot.time_step / total_time, avg_min_dist)

    # if phase == 'test' and config.pas.encoder_type != 'cnn':

    #     avg_occupied_smiliarity = average(total_occupied_similarity)
    #     avg_free_similarity = average(total_free_similarity)
    #     avg_occluded_similarity  = average(total_occluded_similarity)
    #     avg_base_occupied_smiliarity = average(total_base_occupied_similarity)
    #     avg_base_free_similarity = average(total_base_free_similarity)
    #     avg_base_occluded_similarity  = average(total_base_occluded_similarity)
            
    #     avg_similarity = average(total_similarity)
    #     avg_base_similarity = average(total_base_similarity)
        
    #     logging.info(
    #         '{:<5} {}has image similarity(pas/sensor): {:.3f}/{:.3f}'.
    #             format(phase.upper(), extra_info, avg_similarity, avg_base_similarity))
        # if len(similarity) > 1:
        #     logging.info(
        #     ' occupied image similarity(pas/sensor): {:.3f}/{:.3f} and free image similarity(pas/sensor): {:.3f}/{:.3f} and occluded image similarity(pas/sensor): {:.3f}/{:.3f} '.
        #         format(avg_occupied_smiliarity, avg_base_occupied_smiliarity, avg_free_similarity, avg_base_free_similarity, avg_occluded_similarity, avg_base_occluded_similarity))

    logging.info(
        '{:<5} {}has average path length (mean/var): {:.2f}/{:.2f}'.
            format(phase.upper(), extra_info, np.mean(path_lengths) , np.var(path_lengths)))
    if config.action_space.kinematics == 'unicycle':
        chc_total.append(chc)
        logging.info(
        '{:<5} {}has average rotational radius (mean/var): {:.2f}/{:.2f}'.
            format(phase.upper(), extra_info, np.mean(chc_total) , np.var(chc_total)))
        
    logging.info('Collision cases: ' + ' '.join([str(x) for x in collision_cases]))
    logging.info('Timeout cases: ' + ' '.join([str(x) for x in timeout_cases]))

    eval_envs.close()

    print(" Evaluation using {} episodes: mean reward {:.5f}\n".format(
        len(eval_episode_rewards), np.mean(eval_episode_rewards)))
    
    
    return eval_episode_rewards, success_rate
