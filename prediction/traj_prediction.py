import argparse
import pickle
import os
import torch
import numpy as np
from torch.autograd import Variable
from prediction.helper import *
from prediction.model_CollisionGrid import CollisionGridModel


class TrajPrediction():

    def __init__(self, config, cuda=True): 
        # TODO, read the data from argument.py file for the device in train.mppi

        self.obs_length = config.pas.sequence
        self.pred_length = config.diffstack.lookahead_steps
        self.seq_length = self.obs_length + self.pred_length
        # self.method = config.sim.inferred_mathod

        device = torch.device("cuda" if cuda else "cpu")
        self.d = device
        self.timestamp = config.env.time_step # Warning: the predictor should have also been trained with this timestep

        
        # Define the path to the saved args of the model
        save_directory = 'prediction/TrainedModel/CollisionGrid'

        with open(os.path.join(save_directory, 'config.pkl'), 'rb') as f:
            self.saved_args = pickle.load(f)
    
        self.saved_args.device = self.d

        self.net = CollisionGridModel(self.saved_args, infer=True).to(device=self.d)

        # Loading the trained model
        checkpoint_path = os.path.join(save_directory, 'CollisionGrid_model.tar')
        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)
            self.net.load_state_dict(checkpoint['state_dict'])
        else: 
            raise ValueError("No checkpoint found at", checkpoint_path)
        
    
    def ensure_tensor(self, x):
        if not torch.is_tensor(x):
            if isinstance(x, np.ndarray):
                return torch.from_numpy(x)
            else:
                return torch.tensor(x)  # Converts other types (lists, numbers) to a tensor
        return x
    

    def forward(self, x_seq, mask):

        x_seq = self.ensure_tensor(x_seq)
        mask = self.ensure_tensor(mask)

        orig_x_seq = x_seq.clone()

        # grid mask calculation
        grid_seq, grid_TTC_seq = getSequenceInteractionGridMask(x_seq, mask, x_seq, mask,
                                                                self.saved_args.TTC, 
                                                                self.saved_args.D_min, 
                                                                self.saved_args.num_sector,
                                                                self.saved_args.use_cuda)

        x_seq, first_values_dict = position_change(x_seq, mask)
        x_seq = x_seq.to(self.d)

    
        # Extract the observed part of the trajectories
        obs_traj, obs_mask, obs_grid, obs_grid_TTC = x_seq.clone(), mask.clone(), grid_seq.copy(), grid_TTC_seq.copy()
    
        ret_x_seq, dist_param_seq = self.sample(obs_traj, obs_mask, self.net, self.saved_args,
                                                    first_values_dict, orig_x_seq, obs_grid,
                                                    obs_grid_TTC)

        last_obs_frame_mask = mask[-1, :]
        rp_mask = last_obs_frame_mask.unsqueeze(dim=0).repeat(self.pred_length, 1)
        extended_mask = torch.cat((mask, rp_mask), 0)


        ret_x_seq = revert_postion_change(ret_x_seq.cpu(), extended_mask, first_values_dict,
                                               orig_x_seq, self.obs_length, infer=True)


        dist_param_seq[:, :, 0:2] = revert_postion_change(dist_param_seq[:, :, 0:2].cpu(), extended_mask, 
                                                          first_values_dict, orig_x_seq, self.obs_length,
                                                            infer=True)

        return ret_x_seq[self.obs_length:, :, :], dist_param_seq[self.obs_length:, :]

        
    def sample(self, x_seq, mask, net, saved_args, first_values_dict, 
               orig_x_seq, grid, grid_TTC):

        # Number of peds in the sequence
        numx_seq = x_seq.shape[1]

        with torch.no_grad():
            # Construct variables for hidden and cell states
            hidden_states = Variable(torch.zeros(numx_seq, net.args.rnn_size)).to(self.d)
            cell_states = Variable(torch.zeros(numx_seq, net.args.rnn_size)).to(self.d)

        ret_x_seq = Variable(torch.zeros(self.seq_length, numx_seq, x_seq.shape[2])).to(self.d)
        dist_param_seq = Variable(torch.zeros(self.seq_length, numx_seq, 5)).to(self.d)

        # For the observed part of the trajectory
        for tstep in range(self.obs_length-1):

            # Do a forward prop
            grid_t = grid[tstep].to(self.d)
            grid_TTC_t = grid_TTC[tstep].to(self.d)
            out_obs, hidden_states, cell_states = net(x_seq[tstep, :, :].view(1, numx_seq, x_seq.shape[2]), 
                                                        [grid_t], hidden_states, cell_states, 
                                                        mask[tstep, :].view(1, numx_seq), [grid_TTC_t])

            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(out_obs.cpu())

            dist_param_seq[tstep + 1, :, :] = out_obs.clone()

            # # Sample from the bivariate Gaussian
            # next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, mask[tstep, :])
            # ret_x_seq[tstep + 1, :, 0] = next_x
            # ret_x_seq[tstep + 1, :, 1] = next_y

            # Assigning the mean to the next state instead of sampling from the distrbution.
            next_x_mean = mux.clone().data
            next_y_mean = muy.clone().data
            ret_x_seq[tstep + 1, :, 0] = next_x_mean
            ret_x_seq[tstep + 1, :, 1] = next_y_mean

        # Last seen grid
        prev_grid = grid[-1].clone()
        prev_TTC_grid = grid_TTC[-1].clone()

        ret_x_seq[tstep + 1, :, 2:4] = x_seq[-1, :, 2:4]  # vx and vy
        ret_x_seq[tstep + 1, :, 4:] = x_seq[-1,:,4:] # other stored attributes of the humans (gx, gy, r) that will not change from frame to frame
        last_observed_frame_prediction = ret_x_seq[tstep + 1, :, :2].clone()
        ret_x_seq[tstep + 1, :, :2] = x_seq[-1,:,:2] # storing the last GT observed frame here to ensure this is used in the next for loop and then 
        # storing the actual prediction in it after the forward network is run for the first step in the prediction length 

        # in prediction part we continue predictig the trajecotry of those agents that were
        # present in the last timestep of the observation period

        last_obs_frame_mask = mask[-1, :]
        rp_mask = last_obs_frame_mask.unsqueeze(dim=0).repeat(self.pred_length, 1)
        extended_mask = torch.cat((mask, rp_mask), 0)

        # For the predicted part of the trajectory
        for tstep in range(self.obs_length-1, self.pred_length + self.obs_length-1):

            outputs, hidden_states, cell_states = net(ret_x_seq[tstep].view(1, numx_seq, ret_x_seq.shape[2]), 
                                                        [prev_grid], hidden_states, cell_states, 
                                                        last_obs_frame_mask.view(1, numx_seq),
                                                        [prev_TTC_grid])
            if tstep == self.obs_length-1: 
                # storing the actual prediction in the last observed frame position
                ret_x_seq[self.obs_length-1, :, :2] = last_observed_frame_prediction.clone()


            # Extract the mean, std and corr of the bivariate Gaussian
            mux, muy, sx, sy, corr = getCoef(outputs.cpu())

            # Storing the paramteres of the distriution for plotting
            dist_param_seq[tstep + 1, :, :] = outputs.clone()

            # # Sample from the bivariate Gaussian
            # next_x, next_y = sample_gaussian_2d(mux.data, muy.data, sx.data, sy.data, corr.data, last_obs_frame_mask)
            # # Store the predicted position
            # ret_x_seq[tstep + 1, :, 0] = next_x
            # ret_x_seq[tstep + 1, :, 1] = next_y

            # Assigning the mean to the next state instead of sampling from the distrbution.
            next_x_mean = mux.clone().data
            next_y_mean = muy.clone().data
            ret_x_seq[tstep + 1, :, 0] = next_x_mean
            ret_x_seq[tstep + 1, :, 1] = next_y_mean

            # Preparing a ret_x_seq that is covnerted back to the original frame by reverting back to the absolute coordinate.
            # This will be used for grid calculation
            ret_x_seq_convert = ret_x_seq.clone()

            ret_x_seq_convert = revert_postion_change(ret_x_seq_convert.cpu(), extended_mask,
                                                        first_values_dict, orig_x_seq, saved_args.obs_length, infer=True)

            ret_x_seq_convert[tstep + 1, :, 2] = (ret_x_seq_convert[tstep + 1, :, 0] -
                                                ret_x_seq_convert[tstep, :, 0]) / self.timestamp  # vx
            ret_x_seq_convert[tstep + 1, :, 3] = (ret_x_seq_convert[tstep + 1, :, 1] -
                                                ret_x_seq_convert[tstep, :, 1]) / self.timestamp  # vy
            # updating the velocity data in ret_x_seq accordingly
            ret_x_seq[tstep + 1, :, 2] = ret_x_seq_convert[tstep + 1, :, 2].clone()
            ret_x_seq[tstep + 1, :, 3] = ret_x_seq_convert[tstep + 1, :, 3].clone()

            # copy gx, gy, r as it is:
            ret_x_seq[tstep + 1, :, 4:] = x_seq[-1, :, 4:].clone()

            converted_pedlist = [i for i in range(numx_seq) if last_obs_frame_mask[i] == 1]
            list_of_x_seq = Variable(torch.LongTensor(converted_pedlist))
        
            # Get their predicted positions
            current_x_seq = torch.index_select(ret_x_seq_convert[tstep+1], 0, list_of_x_seq)

            prev_grid, prev_TTC_grid = getInteractionGridMask(current_x_seq.data.cpu(), current_x_seq.data.cpu(),
                                                                saved_args.TTC, saved_args.D_min, saved_args.num_sector)

            prev_grid = Variable(torch.from_numpy(prev_grid).float()).to(self.d)
            prev_TTC_grid = Variable(torch.from_numpy(prev_TTC_grid).float()).to(self.d)
                        
        return ret_x_seq, dist_param_seq