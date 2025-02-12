import numpy as np
import torch
import math
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import copy
import pdb

## These are similar to the functions in crowd_sim/envs/crowd_sim_dict.py


def dict_update(humans, robot, config, vis_ids):
    """
        humans: human states at t (batch, K, human_num, 7)
        robot: robot state at t (batch, K, 2)
        
        return: (batch*K, num_agents, features)
    """
    # Assume humans is of shape [batch_size, num_humans, features]
    # and robot is of shape [batch_size, features]
        
    # Flatten batch and K dimensions
    
    humans = humans.view(-1, humans.shape[-2], humans.shape[-1])
    robot = robot.view(-1, robot.shape[-1])

    
    # Preparing tensors for batch processing
    human_pos = humans[:, :, :2]
    # Put dummy values for invisible agents
    if len(vis_ids) > 0:
        invis_ids = [i for i in range(humans.shape[1]) if i not in vis_ids]
        human_pos[:,invis_ids] = torch.tensor([100.,100.], device=human_pos.device).to(human_pos.dtype)
    human_id = torch.arange(humans.shape[1]).repeat(len(human_pos), 1)
    
    robot_pos = robot[:, :2]

    # Creating dictionaries might not be necessary if you can work directly with tensors
    return human_pos, human_id, robot_pos


def disambig_mask(grid_xy, robot_pos, goal, decoded_in_unknown, grid_res, disambig_angle):
    """
    Mask the grid to disambiguate the robot's path towards the goal. 
    The mask is a 120 degree cone towards the goal.
    """
    K = len(robot_pos)
    H, W = grid_xy[0].shape
    # kernel_max = 0.5
    sigma = 0.1
    
    # # # Initialize a grid of coordinates
    # # # TODO: change to 0~knernel_max?
    # # x, y = torch.meshgrid(torch.linspace(-kernel_max, kernel_max, W), torch.linspace(-kernel_max, kernel_max, H))
    # x, y = torch.meshgrid(torch.linspace(-W/2., W/2., W), torch.linspace(-H/2, H/2., H))

    # # # Calculate the distance from the center for each point
    # # r = torch.sqrt(x**2 + y**2)

    # # # Create the bell-shaped kernel
    # # kernel = torch.maximum(torch.tensor(0), kernel_max - r).to(robot_pos.device)
    
    # # # Repeat kernel for batch_size
    # # kernel = kernel.unsqueeze(0).repeat(K, 1, 1)

    # # # Ensure the maximum value is 1 (at the center)
    # # disambig_weight_map = kernel/kernel_max #* 0.0
    
    # kernel = torch.exp(-(x**2 + y**2) / (2 * sigma**2)).unsqueeze(0).repeat(K, 1, 1).to(robot_pos.device)
    # disambig_weight_map = kernel / kernel.sum()
    
    disambig_weight_map = torch.ones((K, H, W), device=robot_pos.device)
    
    # # Obtain rays for the robot towards the goal direction with 120 degree FOV
    heading = torch.atan2(goal[1]-robot_pos[:,1], goal[0]-robot_pos[:,0]) # (K,)
    
    # # Mask points from grid_xy
    x1 = robot_pos[:,0] + 20 * torch.cos(heading-disambig_angle/2.) # (K,)
    y1 = robot_pos[:,1] + 20 * torch.sin(heading-disambig_angle/2.) # (K,)
    
    x2 = robot_pos[:,0] + 20 * torch.cos(heading+disambig_angle/2.) # (K,)
    y2 = robot_pos[:,1] + 20 * torch.sin(heading+disambig_angle/2.) # (K,)
    
    polygon = torch.stack([torch.stack([x1, y1], dim=1), torch.stack([x2, y2], dim=1), robot_pos], dim=1) # (K, 2, 2)
    grid_x = grid_xy[[0]].repeat(K, 1, 1).view(K, -1)
    grid_y = grid_xy[[1]].repeat(K, 1, 1).view(K, -1)
    disambig_mask = pointinpolygon(grid_x,grid_y,polygon.permute(1,0,2)) # (K, H, W)
    disambig_mask = disambig_mask.view(K, H, W)
    
    disambig_weight_map[~disambig_mask] = 0.
    
    # # 4 or 5 polygon points
    # # 2 points from human
    # x1 = r_pos[:,0] + human_radius/torch.tan(theta)*torch.cos(heading-theta) # (K,)
    # y1 = r_pos[:,1] + human_radius/torch.tan(theta)*torch.sin(heading-theta) # (K,)

    # x2 = r_pos[:,0] + human_radius/torch.tan(theta)*torch.cos(heading+theta) # (K,)
    # y2 = r_pos[:,1] + human_radius/torch.tan(theta)*torch.sin(heading+theta) # (K,)
    
    
    # ## TODO: Uncomment to visualize the disambiguating map
    # # self.disambig_reward_grid.append(disambig_weight_map)
    # # self.disambig_reward_grid.append(-torch.abs(torch.ones_like(sensor_grid)*0.5-sensor_grid)+torch.ones_like(sensor_grid)*0.5)
    return disambig_weight_map        



def compute_disambig_cost(robot_pos, grid_xy, sensor_grid, next_sensor_grid, decoded_in_unknown, goal, config):
    """
    For each cell, H(p) = -plogp -(1-p)log(1-p)
    The disambiguation cost is computed on towards the goal direction of the robot with 120 degree cone shaped mask.
    The ambiguation cost is computed on the unobserved area of the sensor grid. 
    The entropy is maximized in the estimated PaS occupancy.
        
    sensor_grid:  [batch_size, H, W] consists of values [0, 0.5, 1] in the same coordinate as decoded_in_unknown.
    decoded_in_unknown: should only be estimating the unobserved areas. Otherwise, it should be the ground truth or zero.        
    """
    # from crowd_sim.envs.generateLabelGrid import generateLabelGrid
    # from crowd_sim.envs.generateSensorGrid import generateSensorGrid
    
    # FOV_radius = config.robot.FOV_radius
    grid_res = config.pas.grid_res
    grid_shape = [config.pas.grid_width, config.pas.grid_height] 
    disambig_method = config.reward.disambig_method
    disambig_angle = config.robot.disambig_angle*torch.pi
    
    disambig_weight_map = disambig_mask(grid_xy,robot_pos, goal, decoded_in_unknown, grid_res, disambig_angle)

    # Calculate the uncertainty reward
    ## We want to compute uncertainty in the unobserved area for both the current and next sensor grid.
    ## Obtain the unobserved area in both current and next sensor grid and set them to 0.5. Otherwise, 0 for both observed free and occupied cells.
    integrated_sensor = torch.zeros(sensor_grid.shape, device=sensor_grid.device).to(decoded_in_unknown.dtype)
    integrated_sensor[torch.logical_and(sensor_grid==0.5, next_sensor_grid==0.5)] = 0.5
    
    ## Transfer the PaS estimation to the unknown area of the integrated unknown grid
    integrated_sensor[integrated_sensor==0.5] = decoded_in_unknown.repeat(len(integrated_sensor),1,1)[integrated_sensor==0.5] 
    zero_mask = integrated_sensor==0.

    integrated_sensor = integrated_sensor * disambig_weight_map * 0.5 # Making the estimated occupancy the most uncertain.
    disambig_reward_map = -integrated_sensor*torch.log(integrated_sensor)-(1-integrated_sensor)*torch.log(1-integrated_sensor) #(~zero_mask) # integrated_sensor == 0. or 1. is nan
    disambig_reward_map[torch.isnan(disambig_reward_map)] = 0. # make nan to 0.
    disambig_H = torch.sum(disambig_reward_map, dim=(1,2))#*coef # (0.03~0.08)*2 torch.sum(disambig_weight_map * disambig_reward_map, dim=(1,2))#*coef # (0.03~0.08)*2

    return disambig_H, disambig_reward_map, disambig_weight_map


def generateLabelGrid(human_pos, h_radius, human_id, robot_pos, res=0.1):  
    import matplotlib.pyplot as plt
    
    # Parameters
    grid_half_size = 5.0  # Half width/height of the grid
    
    # Compute coordinates based on ego positions
    # Create ranges for x and y coordinates centered around zero
    base_x_coords = torch.arange(-grid_half_size, grid_half_size, step=res, device=robot_pos.device)
    base_y_coords = torch.arange(-grid_half_size, grid_half_size, step=res, device=robot_pos.device)
    
    # Meshgrid of these coordinates
    base_mesh_x, base_mesh_y = torch.meshgrid(base_x_coords, base_y_coords, indexing='xy')
    
    # Adding batch dimension and expanding to match the batch size
    base_mesh_x = base_mesh_x.expand(robot_pos.shape[0], -1, -1)
    base_mesh_y = base_mesh_y.expand(robot_pos.shape[0], -1, -1)
    
    # Offset these coordinates by the ego positions to center them around the ego
    x_local = base_mesh_x + robot_pos[:, 0].unsqueeze(1).unsqueeze(2)
    y_local = base_mesh_y + robot_pos[:, 1].unsqueeze(1).unsqueeze(2)
                
    # Initialize label grid
    label_grid = torch.full((robot_pos.shape[0], 2, base_mesh_x.shape[1], base_mesh_x.shape[2]), torch.tensor(0.), device=robot_pos.device)
    label_grid[:, 1, :, :] = torch.nan  # For unoccupied cells, they remain NaN
    
    # Compute distances and apply sensor radius mask
    sensor_positions = human_pos.unsqueeze(2).unsqueeze(3)  # [K, num_humans, 1, 1, 2]
    sensor_radii = h_radius.unsqueeze(2).unsqueeze(3)  # [K, num_humans, 1, 1]
    
    distances = torch.sqrt((x_local.unsqueeze(1) - sensor_positions[..., 0])**2 + (y_local.unsqueeze(1) - sensor_positions[..., 1])**2)
    mask = distances < sensor_radii  # [K, num_humans, height, width]
        
    # Vectorized setting of occupied cells and sensor IDs
    batch_indices, sensor_indices, height_indices, width_indices = mask.nonzero(as_tuple=True)
    label_grid[batch_indices, 0, height_indices, width_indices] = 1
    label_grid[batch_indices, 1, height_indices, width_indices] = sensor_indices.float()
    
    return label_grid, x_local, y_local


# Find the unknown cells using polygons for faster computation. (Result similar to ray tracing)
def generateSensorGrid(label_grid, h_pos, h_radius, r_pos, map_xy, FOV_radius, res=0.1):
    """
    label_grid: [K, 2, H, W]
    human_pos: [K, num_humans, 2]
    h_radius: [K, num_humans]
    robot_pos: [K, 2]
    map_xy: [[K, H, W], [K, H, W]]
    FOV_radius: float
    """
    K, H, W = map_xy[0].shape
    num_h = h_pos.shape[-2]
    x_local, y_local = map_xy 
    
    h_id = torch.arange(num_h, device = h_pos.device).repeat(h_pos.shape[0], 1)

    # center_ego = ego_dict['pos']
    occluded_id = torch.ones((K,num_h), device=h_pos.device)*-1
    visible_id = []

    # # get the maximum and minimum x and y values in the local grids
    # x_shape = x_local.shape[-2]
    # y_shape = x_local.shape[-1]	

    id_grid = label_grid[:,1].clone() # [K,H,W]
    id_flat = id_grid.view(K, -1)
    id_flat = torch.where(torch.isnan(id_flat), torch.tensor(-1., device=id_flat.device), id_flat)
    unique_id = [torch.unique(row) for row in id_flat] # does not include ego (robot) id
    # padding with -1 to make the length of each row same
    from torch.nn.utils.rnn import pad_sequence

    # Assuming unique_id is a list of tensors
    # First pad all tensors to the length of the largest tensor
    padded = pad_sequence(unique_id, batch_first=True, padding_value=-1)

    # Now, truncate or further pad each row to the desired length num_h
    # padded_unique_id = padded[:, :num_h]  # Truncate
    if padded.size(1) < num_h:
        # If needed, pad to ensure all rows have exactly num_h columns
        padding_size = num_h - padded.size(1)
        padding_tensor = torch.full((padded.size(0), padding_size), -1, dtype=padded.dtype).to(unique_id[0].device)
        padded_unique_id = torch.cat([padded, padding_tensor], dim=1)    
    else:
        padded_unique_id = padded
        
    # torch.unique(id_flat, dim=-1) won't work as intended as it will find unique in column-wise manner
    # unique_per_row_padded = torch.nn.utils.rnn.pad_sequence(unique_per_row, batch_first=True, padding_value=-1)

    # # cells not occupied by ego itself
    # mask = torch.where(label_grid[0]!=2, True,False)

    # # no need to do ray tracing if no object on the grid
    if torch.all(label_grid[:,0]==0.):
        sensor_grid = torch.zeros(x_local.shape, device=x_local.device) # [K, H, W]

    else:
        sensor_grid = torch.zeros(x_local.shape, device=x_local.device) 

        # ref_pos = np.array(ref_dict['pos'])
        # ref_r = np.array(ref_dict['r'])

        # Find the cells that are occluded by the obstructing human agents
        # reorder humans according to their distance from the robot.
        distance = torch.linalg.norm(r_pos.unsqueeze(1)-h_pos, dim=-1) # (K, num_humans)
        sort_indx = torch.argsort(distance, dim=-1) # (K, num_humans)
        unchecked_id = sort_indx.to(h_id.dtype) #h_id.gather(1, sort_indx)
        # Create occlusion polygons starting from closest humans. Reject humans that are already inside the polygons.
        
        # Reshape to (n_humans, ..)
        h_pos= h_pos.gather(1, sort_indx.unsqueeze(-1).repeat(1,1,2)).permute(1,0,2) # (num_humans, K, 2)
        h_radius = h_radius.permute(1,0) # (num_humans, K)
        unchecked_id = unchecked_id.permute(1,0)#[[:2]] # (num_humans, K)
        unchecked_id2 = unchecked_id.clone()
        
        for center, human_radius, id in zip(h_pos, h_radius, unchecked_id):	
            """
            center: [K, 2]
            human_radius: [K]
            id: [K]
            """
            # if human is already occluded, use dummy value
            dummy_mask = torch.any(torch.isin(occluded_id, id),dim=1)

            id[dummy_mask] = -1.
            
            ## Check with distance = torch.randint(0,5,(3,3)); id = torch.tensor([2,3,1]); mask = distance==id.reshape(3,1) 
            hmask = label_grid[:,1]==id.reshape(len(id),1,1)
            sensor_grid[hmask] = 1.

            
            alpha = torch.atan2(center[:,1]-r_pos[:,1], center[:,0]-r_pos[:,0]) # (K,)
            theta = torch.asin(torch.clip(human_radius/torch.sqrt((center[:,1]-r_pos[:,1])**2 + (center[:,0]-r_pos[:,0])**2), -1., 1.)) # (K,)
            
            # 4 or 5 polygon points
            # 2 points from human
            x1 = r_pos[:,0] + human_radius/torch.tan(theta)*torch.cos(alpha-theta) # (K,)
            y1 = r_pos[:,1] + human_radius/torch.tan(theta)*torch.sin(alpha-theta) # (K,)

            x2 = r_pos[:,0] + human_radius/torch.tan(theta)*torch.cos(alpha+theta) # (K,)
            y2 = r_pos[:,1] + human_radius/torch.tan(theta)*torch.sin(alpha+theta) # (K,)

            # Choose points big/small enough to cover the region of interest in the grid
            x3 = torch.where(x1 <= r_pos[:,0], -12., 12.) # (K,)
            y3 = linefunction(r_pos[:,0],r_pos[:,1],x1,y1,x3) # (K,)
            x4 = torch.where(x2 <= r_pos[:,0], -12., 12.) # (K,)
            y4 = linefunction(r_pos[:,0],r_pos[:,1],x2,y2,x4) # (K,)
            
            # For the dummy make the occlusion mask out side of the grid to ignore.
            x1[dummy_mask] = 13.
            x2[dummy_mask] = 13.
            x3[dummy_mask] = 13.
            x4[dummy_mask] = 13.
            y1[dummy_mask] = 13.
            y2[dummy_mask] = 13.
            y3[dummy_mask] = 13.
            y4[dummy_mask] = 13.
            
            # if x1 <= r_pos[:,0]:
            #     x3 = -12. 
            # else:
            #     x3 = 12. 
            # y3 = linefunction(r_pos[:,0],r_pos[:,1],x1,y1,x3)
            # if x2 <= r_pos[:,0]:
            #     x4 = -12. 
            # else:
            #     x4 = 12. 
            # y4 = linefunction(r_pos[:,0],r_pos[:,1],x2,y2,x4)
            grid_points = torch.stack([x_local.view(K, -1), y_local.view(K, -1)], dim=-1) # (K, H*W, 2)
            polygon_points = torch.stack([torch.stack([x1, y1], dim=1), torch.stack([x2, y2], dim=1), torch.stack([x4, y4], dim=1), torch.stack([x3, y3], dim=1)], dim=1) # (K, 4, 2)

            occ_mask = parallelpointinpolygon(grid_points, polygon_points) # occ_mask: (K, H*W)
            
            occ_mask = occ_mask.reshape(x_local.shape)
            sensor_grid[occ_mask] = 0.5

            # check if any agent is fully inside the polygon (occluded area)
            for oid in unchecked_id: # oid : (K,), unchecked_id : (num_humans, K)
                # get the mask of the agent
                agent_mask = (label_grid[:,1,:,:]==oid.unsqueeze(1).unsqueeze(2))
                # if any agent is fully inside the polygon store in the occluded_id and opt from unchecked_id	
                
                # mask only the fully occluded agents
                occ1_mask = sensor_grid*agent_mask  # (K, H, W)
                # count the number of occluded cells
                agent_occ = torch.sum(occ1_mask==0.5, dim=(1,2)) # (K,)
                # check if the number of occluded cells is equal to the number of cells of the agent
                occ_id_mask = agent_occ == torch.sum(agent_mask, dim=(1,2)) # (K,)

                if torch.any(occ_id_mask): #torch.any(occ_id_mask):
                    # turn occ_id_mask to indices
                    # save the occluded id (oid) in occluded_id to filter out in the next loop
                    occluded_id[torch.arange(occluded_id.size(0)), oid] = torch.where(occ_id_mask, oid, occluded_id[torch.arange(occluded_id.size(0)), oid])                 
                    # add a dummy value unchecked_id if it is checked to avoid the shape mismatch
                    # no need to check this id again in the next loop as they are already occluded
                    
                    # mark the occluded agent as -1 in the unchecked_id to avoid checking again
                    unchecked_id2[oid,torch.arange(unchecked_id2.size(1))] = torch.where(occ_id_mask, -1, unchecked_id2[oid,torch.arange(unchecked_id2.size(1))]) # TODO: check if oid is correct, instead of h_id

                unchecked_id2 = torch.where(unchecked_id2==id, -1, unchecked_id2)


    # # Set cells out side of field of view as unknown
    FOVmask = point_in_circle(x_local, y_local, r_pos, FOV_radius, res) 
    sensor_grid[~FOVmask] = 0.5
    

    # padded_unique_id: (K, num_humans)
    for id in padded_unique_id.T:
        mask_agent = (label_grid[:,1]==id.unsqueeze(1).unsqueeze(2)) # (K, H, W)
        vis_mask1 = sensor_grid * mask_agent.to(sensor_grid.dtype) # (K, H, W) # check if there are visible cells (1) for the agent
        vis_mask2 = torch.any(vis_mask1.view(K,-1)==1., dim=1) # (K,) # check if there are any partial visible cells for the agent
        vis_mask = mask_agent*vis_mask2.unsqueeze(1).unsqueeze(2) # agent with partial visible cells

        sensor_grid[vis_mask] = 1. 
        
        # visible_id.append(id)

    return sensor_grid# , visible_id 


def global_grid(origin,endpoint,res):
    # create a grid of x and y values that have an associated x,y centre
    # coordinate in the global space (42.7 m buffer around the edge points)
    ### myj : don't need to consider the buffer in our scenario
    xmin = min(origin[0],endpoint[0]) #+ res/2. #- 128./2. # 2.*(128./3.)
    xmax = max(origin[0],endpoint[0]) + res/2.#+ 128./2. # 2.*(128./3.) # + res
    ymin = min(origin[1],endpoint[1]) #+ res/2.#- 128./2  # 2.*(128./3.)
    ymax = max(origin[1],endpoint[1]) + res/2. #+ 128./2. # 2.*(128./3.) # + res

    x_coords = torch.arange(xmin,xmax,res)
    y_coords = torch.arange(ymin,ymax,res)

    gridx,gridy = torch.meshgrid(x_coords,y_coords)
    # TODO: change to torch.flipud(gridy): does not really affect the result or visualization though.
    return gridx, gridy # torch.flipud(gridy) in order to make the values increasing from bottom to top


# TODO: There should be a better way to do this in Pytorch
# TODO: indice matching should be done in a batched manner
# def transfer_grid_data(curr_xy, curr_grids, next_xy, next_grids, distance_threshold=0.1, batch_chunk_size=1):
#     """
#     Transfers data from curr_grids to next_grids based on coordinate mapping
#     from curr_xy (ego grid) to next_xy (map grid) only if the coordinates are within
#     a specified distance threshold.
    
    
#     Arguments:
#     curr_xy = [[B,H,W], [B,H,W]]   
#     curr_grids = [B,H,W] 
#     next_xy = [[H,W], [H,W]]
#     next_grids = [H,W]
#     distance_threshold -- A float specifying the maximum distance for considering two points a match.

#     Returns:
#     Updated grids with the transferred data from next_grids.
#     """
#     B, H, W = curr_grids.shape
    
#     # Initialize result tensor for updated pred_maps
#     updated_grids = torch.ones_like(curr_grids).cuda()*0.5
    
#     next_xy_stacked = next_xy.permute(1,2, 0).to(curr_grids.dtype)  # (H, W, 2)
#     curr_xy_stacked = torch.stack(curr_xy).permute(1,2,3,0).to(curr_grids.dtype)  # (B, H, W, 2)

#     # # Compute pairwise distances between next_xy and curr_xy
#     # # (H*W, B, 2)-(H*W, 1, 2) = (H*W, B, 1)
    
#     # dists = torch.cdist(curr_xy_stacked.view(B, -1,2).permute(1,0,2),next_xy_stacked.view(-1, 2) .unsqueeze(1), p=2)  # (H*W, B, 1)
#     next_xy_flat = next_xy_stacked.view(-1, 2)  # (H*W, 2)
#     kdtree = cKDTree(next_xy_flat.cpu().numpy())
        
#     # Process the batches in chunks to avoid memory overflow
#     for start in range(0, B, batch_chunk_size):
#         end = min(start + batch_chunk_size, B)

#         # Extract the chunk of batches
#         curr_grids_chunk = curr_grids[start:end]  # Shape: (batch_chunk_size, H_map, W_map)
#         curr_xy_chunk = curr_xy_stacked[start:end]  # Shape: (batch_chunk_size, H_map, W_map, 2)
#         curr_xy_chunk_flat = curr_xy_chunk.view(end-start, -1, 2)  # (batch_chunk_size, H_map*W_map, 2)

#         # Flatten the grid coordinates for efficient index mapping
#         curr_xy_flat = curr_xy_chunk.view(end-start, -1, 2)  # (batch_chunk_size, H_map*W_map, 2)
#         # next_grids_flat = next_grids.repeat(end-start, 1, 1).view(-1)  # Flatten next grid data to (H*W)
                            
#         ## Memory issue;;
#         # Compute pairwise distances between next_xy and curr_xy
#         #(B, H*W, 1, 2)-(1,1, H*W, 2) = (B, H*W, H*W, 2)
#         # dists = (curr_xy_flat.unsqueeze(2) - next_xy_flat.unsqueeze(0).unsqueeze(0)).norm(dim=-1)  # (batch_chunk_size, H*W, H*W)
#         # ( B, H*W, 2) - (1, H*W, 2)
#         # dists = torch.cdist(curr_xy_chunk_flat.unsqueeze(2), next_xy_flat.unsqueeze(0).unsqueeze(0).repeat(len(curr_xy_chunk_flat), 1, 1,1), p=2).view(batch_chunk_size, H*W, -1)  # (batch_chunk_size, H_map*W_map, H*W)
#         # # Find the closest match in curr_grids for each point in curr_grids
#         # min_dists, min_dist_indices = dists.min(dim=-1)  # (batch_chunk_size, H*W)
        
#         distances, indices = kdtree.query(curr_xy_chunk_flat.cpu().numpy()[0], distance_upper_bound=distance_threshold)  # (batch_chunk_size, H*W)
#         mask = distances <= distance_threshold 
#         mask = torch.from_numpy(mask).cuda()
#         indices = torch.from_numpy(indices).cuda()

#         # Using the mask, transfer the values in parallel
#         curr_grids_flat = curr_grids_chunk.view(end-start, -1) # Flatten curr_grids to (batch_chunk_size, H_map*W_map)

#         next_indices = indices[mask]
#         curr_indices = torch.arange(H*W).to(indices)[mask]
        
#         # updated_grids[start:end].view(end-start, -1)[next_indices.unsqueeze(0)] = curr_grids_flat[curr_indices.unsqueeze(0)]
#         updated_grids[start:end].view(-1)[next_indices] = curr_grids_flat.view(-1)[curr_indices]

#     return updated_grids

### Using keops: https://github.com/getkeops/keops/tree/main
def transfer_grid_data(curr_xy, curr_grids, next_xy, next_grids, distance_threshold=0.1, batch_chunk_size=1):
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
    B, H, W = curr_grids.shape
    
    # Initialize result tensor for updated pred_maps
    updated_grids = torch.ones_like(curr_grids).cuda()*0.5
    
    next_xy_stacked = next_xy.permute(1,2, 0).to(curr_grids.dtype)  # (H, W, 2)
    curr_xy_stacked = torch.stack(curr_xy).permute(1,2,3,0).to(curr_grids.dtype)  # (B, H, W, 2)

    # # Compute pairwise distances between next_xy and curr_xy
    # # (H*W, B, 2)-(H*W, 1, 2) = (H*W, B, 1)
    
    # dists = torch.cdist(curr_xy_stacked.view(B, -1,2).permute(1,0,2),next_xy_stacked.view(-1, 2) .unsqueeze(1), p=2)  # (H*W, B, 1)
    next_xy_flat = next_xy_stacked.view(-1, 2)  # (H*W, 2)
    kdtree = cKDTree(next_xy_flat.cpu().numpy())
        
    # Process the batches in chunks to avoid memory overflow
    for start in range(0, B, batch_chunk_size):
        end = min(start + batch_chunk_size, B)

        # Extract the chunk of batches
        curr_grids_chunk = curr_grids[start:end]  # Shape: (batch_chunk_size, H_map, W_map)
        curr_xy_chunk = curr_xy_stacked[start:end]  # Shape: (batch_chunk_size, H_map, W_map, 2)
        curr_xy_chunk_flat = curr_xy_chunk.view(end-start, -1, 2)  # (batch_chunk_size, H_map*W_map, 2)

        # Flatten the grid coordinates for efficient index mapping
        curr_xy_flat = curr_xy_chunk.view(end-start, -1, 2)  # (batch_chunk_size, H_map*W_map, 2)
        # next_grids_flat = next_grids.repeat(end-start, 1, 1).view(-1)  # Flatten next grid data to (H*W)
        
        # from pykeops.torch import LazyTensor
        # curr_xy_chunk_flat = LazyTensor(curr_xy_chunk_flat.unsqueeze(2))  # (batch_chunk_size, H_map*W_map, 1, 2)
        # next_xy_flat = LazyTensor(next_xy_flat.unsqueeze(0).unsqueeze(0))  # (1, 1, H*W, 2)
        # pdb.set_trace()
                            
        # ## Memory issue;;
        # # Compute pairwise distances between next_xy and curr_xy
        # #(B, H*W, 1, 2)-(1,1, H*W, 2) = (B, H*W, H*W, 2)
        # dists = (curr_xy_flat.unsqueeze(2) - next_xy_flat.unsqueeze(0).unsqueeze(0)).norm(dim=-1)  # (batch_chunk_size, H*W, H*W)
        # # ( B, H*W, 2) - (1, H*W, 2)
        # dists = torch.cdist(curr_xy_chunk_flat.unsqueeze(2), next_xy_flat.unsqueeze(0).unsqueeze(0).repeat(len(curr_xy_chunk_flat), 1, 1,1), p=2).view(batch_chunk_size, H*W, -1)  # (batch_chunk_size, H_map*W_map, H*W)
        # # Find the closest match in curr_grids for each point in curr_grids
        # min_dists, min_dist_indices = dists.min(dim=-1)  # (batch_chunk_size, H*W)
        # mask = min_dists <= distance_threshold
        
        # # Using the mask, transfer the values in parallel
        # curr_grids_flat = curr_grids_chunk.view(end-start, -1) # Flatten curr_grids to (batch_chunk_size, H_map*W_map)
        
        # next_indices = min_dist_indices[mask]
        # curr_indices = torch.arange(H*W).to(min_dist_indices)[mask]
        
        
        distances, indices = kdtree.query(curr_xy_chunk_flat.cpu().numpy()[0], distance_upper_bound=distance_threshold)  # (batch_chunk_size, H*W)
        mask = distances <= distance_threshold 
        mask = torch.from_numpy(mask).cuda()
        indices = torch.from_numpy(indices).cuda()

        # Using the mask, transfer the values in parallel
        curr_grids_flat = curr_grids_chunk.view(end-start, -1) # Flatten curr_grids to (batch_chunk_size, H_map*W_map)

        next_indices = indices[mask]
        curr_indices = torch.arange(H*W).to(indices)[mask]
        
        # updated_grids[start:end].view(end-start, -1)[next_indices.unsqueeze(0)] = curr_grids_flat[curr_indices.unsqueeze(0)]
        updated_grids[start:end].view(-1)[next_indices] = curr_grids_flat.view(-1)[curr_indices]

    return updated_grids




def Transfer_to_EgoGrid(curr_xy, curr_grids, next_xy, next_grids, res):
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
    global_grid_x, global_grid_y = global_grid(torch.tensor([x_min,y_min]),torch.tensor([x_max,y_max]),global_res)

    x_min = torch.min(global_grid_x)
    x_max = torch.max(global_grid_x)
    y_min = torch.min(global_grid_y)
    y_max = torch.max(global_grid_y)
    
    curr_grids_ = curr_grids.clone()
    updated_next_grids = transfer_grid_data(curr_xy, curr_grids_, next_xy, next_grids ,distance_threshold=global_res)

    return updated_next_grids



def point_in_circle_single_sample(x_local, y_local, center, radius, res):
    mask = torch.sqrt((x_local-center[0])**2+(y_local-center[1])**2) < radius  
    return mask

def point_in_circle(x_local, y_local, center, radius, res):
    mask = torch.sqrt((x_local-center[:,0].reshape(-1,1,1))**2+(y_local-center[:,1].reshape(-1,1,1))**2) < radius  
    return mask
# generate the y indeces along a line
def linefunction(velx,vely,indx,indy,x_range):
    m = (indy-vely)/(indx-velx) 
    b = vely-m*velx
    return m*x_range + b 

def pointinpolygon_single_sample(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in range(n+1):
        p2x,p2y = poly[i % n]
        if y > min(p1y,p2y):
            if y <= max(p1y,p2y):
                if x <= max(p1x,p2x):
                    if p1y != p2y:
                        xints = (y-p1y)*(p2x-p1x)/(p2y-p1y)+p1x
                    if p1x == p2x or x <= xints:
                        inside = not inside
        p1x,p1y = p2x,p2y

    return inside

def parallelpointinpolygon_single_sample(points, polygon):
    """
    inputs:
    - points: (K, 2, H*W) tensor of floats, where points[i,j] is the j-th point of the i-th batch
    - polygon: (K, 4, 2) tensor of floats, where polygon[i,j] is the j-th vertex of the i-th polygon
    return: D = (K, H*W) tensor of bools, where D[i,j] is True if points[i] is inside polygon[j]
    """
    D = torch.empty(len(points), dtype=torch.bool) 
    for i in range(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D   

def pointinpolygon(x,y,poly):
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
    poly = poly.to(x.dtype)
    x = x#.to(poly.dtype)
    y = y#.to(poly.dtype)
    # poly = poly#.to(poly.dtype)
    
    inside = torch.zeros((K, HW), device=x.device, dtype=torch.bool) # (K, H*W)
    p2x = torch.zeros((K,1), device=x.device)
    p2y = torch.zeros((K,1), device=x.device)
    xints = torch.zeros(x.shape, device=x.device, dtype=x.dtype)
    p1x,p1y = poly[0,:,[0]], poly[0,:,[1]] # (K,1), (K,1)
    
    for i in range(1,n+1):
        p2x,p2y = poly[i % n,:,[0]], poly[i % n,:,[1]] # (K,1), (K,1)
        mask = (y> torch.min(p1y, p2y)) & (y <= torch.max(p1y, p2y)) & (x <= torch.max(p1x, p2x)) # (K, H*W)
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



def parallelpointinpolygon(points, polygon):
    """
    inputs:
    - points: (K, H*W, 2) tensor of floats, where points[i,j] is the j-th point of the i-th batch
    - polygon: (K, 4, 2) tensor of floats, where polygon[i,j] is the j-th vertex of the i-th polygon
    return: D = (K, H*W) tensor of bools, where D[i,j] is True if points[i] is inside polygon[j]
    """
    # D = torch.empty(points.shape[:2], dtype=torch.bool)
    # for i in range(0, len(D)):
    D = pointinpolygon(points[:,:,0], points[:,:,1], polygon.permute(1,0,2) )
    return D    





