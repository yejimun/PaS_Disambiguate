import numpy as np
import torch
import matplotlib.pyplot as plt
import pdb

## These are similar to the functions in crowd_sim/envs/crowd_sim_dict.py

def dict_update(humans, robot, config):
    """[summary]
    Updates the current state dictionary (robot_dict, human_dict)
    For creating label/sensor grid
    """
    # only need pose, r, id (order does not matter here for the planner)
    # myj: use the same radius humans for now
    h_radius = humans[:,-1] #config.humans.radius
    r_radius = robot[-1] #config.robot.radius
    
    human_id = []
    human_pos = []
    human_radius = []

    for i in range(len(humans)):
        human_pos.append(torch.Tensor([humans[i,0], humans[i,1]]).to(humans.device))
        human_radius.append(h_radius[i]) # h_radius)
        human_id.append(i)


    robot_pos = torch.Tensor([robot[0],robot[1]]).to(robot.device)

    keys = ['id','pos', 'r']
    robot_values = [100, robot_pos, r_radius]
    robot_dict = dict(zip(keys, robot_values))

    human_values = [human_id, human_pos, human_radius]
    humans_dict = dict(zip(keys, human_values))
    
    return robot_dict, humans_dict


def disambig_mask(grid_shape, robot_pos, goal):
    # creates a bell-shaped 50x50 kernel that has 1 at the center of the grid, but 0 at the most other edge of the grid.
    # This kernel is used to calculate the uncertainty reward
    # The uncertainty reward is calculated by multiplying the kernel with the observation grid
    """Creates a bell-shaped kernel."""
    # Initialize a grid of coordinates
    x, y = torch.meshgrid(torch.linspace(-1, 1, grid_shape[0]), torch.linspace(-1, 1, grid_shape[1]))
    
    # Calculate the distance from the center for each point
    r = torch.sqrt(x**2 + y**2)

    # Create the bell-shaped kernel
    kernel = torch.maximum(torch.tensor(0), 1 - r).to(robot_pos.device)

    # Ensure the maximum value is 1 (at the center)
    disambig_reward_map = kernel/torch.max(kernel)

    ## TODO: Uncomment to train with disambiguating reward towards goal & comment out /4. in the return line
    # Mask only the quarter of the map that is from the robot to the goal
    if robot_pos[0] < goal[0]:
        disambig_reward_map[:, :int(grid_shape[1] / 2)] = 0
    else:
        disambig_reward_map[:, int(grid_shape[1] / 2):] = 0
        
    if robot_pos[1] < goal[1]:
        disambig_reward_map[:int(grid_shape[0] / 2), :] = 0
    else:
        disambig_reward_map[int(grid_shape[0] / 2):, :] = 0

    ## TODO: Uncomment to visualize the disambiguating map
    # self.disambig_reward_grid.append(disambig_reward_map)
    # self.disambig_reward_grid.append(-torch.abs(torch.ones_like(sensor_grid)*0.5-sensor_grid)+torch.ones_like(sensor_grid)*0.5)
    return disambig_reward_map        
    


def compute_disambig_cost(gt_neighbors, x, goal, config):
    # from crowd_sim.envs.generateLabelGrid import generateLabelGrid
    # from crowd_sim.envs.generateSensorGrid import generateSensorGrid
    
    FOV_radius = config.robot.FOV_radius
    grid_res = config.pas.grid_res
    grid_shape = [config.pas.grid_width, config.pas.grid_height] 
    disambig_method = config.reward.disambig_method
    
    # TODO: choose different timesteps to compute disambig_cost
    # compute disambig_cost for the last timestep 
    
    ego_dict, other_dict = dict_update(gt_neighbors, x, config)
    label_grid, x_local, y_local = generateLabelGrid(ego_dict, other_dict)
    map_xy = [x_local, y_local]
    visible_id, sensor_grid = generateSensorGrid(label_grid, ego_dict, other_dict, map_xy, FOV_radius, res=grid_res)
    
    disambig_reward_map = disambig_mask(grid_shape, x, goal)

    # Calculate the uncertainty reward
    ## TODO: examine the effect of the uncertainty reward
    ones_grid = torch.ones_like(sensor_grid).to(x.device)
    if disambig_method =='linear':
        reward = torch.sum(disambig_reward_map * (-torch.abs(ones_grid*0.5-sensor_grid)+ones_grid*0.5))/(grid_shape[0]*grid_shape[1]/4)*2/4. # 0.08~0.13
    elif disambig_method == 'entropy':
        coef = 1/(grid_shape[0]*grid_shape[1]/4)*1/2.
        # coef = 1.
        epsilon = 1e-10
        reward = torch.sum(disambig_reward_map * (-sensor_grid*torch.log(sensor_grid+epsilon)-(1-sensor_grid)*torch.log(1-sensor_grid+epsilon)))*coef # (0.03~0.08)*2
        # print("disambig",reward)
        # print(disambig_reward_map, (-sensor_grid*torch.log(sensor_grid)-(1-sensor_grid)*torch.log(1-sensor_grid)), coef)
    else:
        raise NotImplementedError
    return reward


def generateLabelGrid(ego_dict, sensor_dict, res=0.1):  
    import matplotlib.pyplot as plt

    minx = ego_dict['pos'][0] - 5. + res/2.
    miny = ego_dict['pos'][1] - 5. + res/2. 
    maxx = ego_dict['pos'][0] + 5.  
    maxy = ego_dict['pos'][1] + 5. 

    x_coords = torch.arange(minx,maxx,res)
    y_coords = torch.arange(miny,maxy,res)

    mesh_x, mesh_y = torch.meshgrid(x_coords,y_coords)
    x_local = mesh_x.to(ego_dict['pos'].device)
    y_local = mesh_y.to(ego_dict['pos'].device)

    label_grid = torch.zeros((2,x_local.shape[0],x_local.shape[1])).to(ego_dict['pos'].device) 
    label_grid[1] = torch.nan # For unoccupied cell, they remain nan
    
    # Optimization: Compute all distances in a batch operation instead of point_in_circle()
    sensor_positions = torch.stack(sensor_dict['pos'])  # Shape: [num_sensors, 2]
    sensor_radii = torch.tensor(sensor_dict['r']).unsqueeze(-1).unsqueeze(-1).to(ego_dict['pos'].device)  # Shape: [num_sensors, 1, 1]
    distances = torch.sqrt((x_local - sensor_positions[:, 0, None, None])**2 + (y_local - sensor_positions[:, 1, None, None])**2).to(ego_dict['pos'].device) 
    mask = distances < sensor_radii  # Shape: [num_sensors, height, width]
    
    if torch.any(mask):
        for i, s_id in enumerate(sensor_dict['id']):
            # mask = point_in_circle(x_local, y_local, pos, radius, res)

            # occupied by sensor
            label_grid[0,mask[i]] = 1. 
            label_grid[1,mask[i]] = int(s_id) # does not include ego id
            
    ## plot and save the label_grid[0] and label_grid[1] to see the grid
    # plt.imshow(label_grid[0].cpu().numpy())
    # plt.imshow(label_grid[1].cpu().numpy())
    # plt.savefig('label_grid.png')

    return label_grid, x_local, y_local


def generateSensorGrid(label_grid, ego_dict, ref_dict, map_xy, FOV_radius, res=0.1):
    x_local, y_local = map_xy

    center_ego = ego_dict['pos']
    occluded_id = []
    visible_id = []

    # get the maximum and minimum x and y values in the local grids
    x_shape = x_local.shape[0]
    y_shape = x_local.shape[1]	

    id_grid = label_grid[1].clone()


    unique_id = torch.unique(id_grid) # does not include ego (robot) id

    # cells not occupied by ego itself
    # mask = torch.where(label_grid[1]!=100, True,False)

    # no need to do ray tracing if no object on the grid
    if torch.all(label_grid[0]==0.):
        sensor_grid = torch.zeros((x_shape, y_shape)).to(ego_dict['pos'].device)

    else:
        sensor_grid = torch.zeros((x_shape, y_shape)).to(ego_dict['pos'].device)
        ref_pos = torch.stack(ref_dict['pos'])
        ref_r = torch.Tensor(ref_dict['r'])

        # Find the cells that are occluded by the obstructing human agents
        # reorder humans according to their distance from the robot.
        # distance = torch.Tensor([torch.linalg.norm(center-center_ego) for center in ref_pos])
        distance = torch.norm(ref_pos - ego_dict['pos'], dim=1)
        sort_indx = torch.argsort(distance)

        unchecked_id = torch.Tensor(ref_dict['id'])[sort_indx]
        
        ## check only the humans that are within the FOV
        unchecked_id = unchecked_id[torch.where(distance[sort_indx] < FOV_radius+ref_r[0])[0]].long()
        
        # Create occlusion polygons starting from closest humans. Reject humans that are already inside the polygons.
        for h_id in unchecked_id:	
            # if human is already occluded, then just pass            
            center = ref_pos[h_id]
            human_radius = ref_r[h_id]
            
            if h_id in occluded_id:
                continue

            hmask = (label_grid[1,:,:]==h_id)
            sensor_grid[hmask] = 1.

            alpha = torch.atan2(center[1]-center_ego[1], center[0]-center_ego[0])
            theta = torch.asin(torch.clip(human_radius/torch.sqrt((center[1]-center_ego[1])**2 + (center[0]-center_ego[0])**2), -1., 1.))
            
            # 4 or 5 polygon points
            # 2 points from human
            x1 = center_ego[0] + human_radius/torch.tan(theta)*torch.cos(alpha-theta)
            y1 = center_ego[1] + human_radius/torch.tan(theta)*torch.sin(alpha-theta)

            x2 = center_ego[0] + human_radius/torch.tan(theta)*torch.cos(alpha+theta)
            y2 = center_ego[1] + human_radius/torch.tan(theta)*torch.sin(alpha+theta)

            # Choose points big/small enough to cover the region of interest in the grid
            if x1 <= center_ego[0]:
                x3 = -12. 
            else:
                x3 = 12. 
            y3 = linefunction(center_ego[0],center_ego[1],x1,y1,x3)
            if x2 <= center_ego[0]:
                x4 = -12. 
            else:
                x4 = 12. 
            y4 = linefunction(center_ego[0],center_ego[1],x2,y2,x4)

            polygon_points = torch.Tensor([[x1, y1], [x2, y2], [x4, y4],[x3, y3]]).to(ego_dict['pos'].device)
            grid_points = torch.stack([x_local.flatten(), y_local.flatten()])	


            occ_mask = parallelpointinpolygon(grid_points.T, polygon_points)
            occ_mask = occ_mask.reshape(x_local.shape)
            sensor_grid[occ_mask] = 0.5

            # check if any agent is fully inside the polygon
            for oid in unchecked_id:
                oid_mask = (label_grid[1,:,:]==oid)
                # if any agent is fully inside the polygon store in the occluded_id and opt from unchecked_id			
                if torch.all(sensor_grid[oid_mask] == 0.5):
                    occluded_id.append(oid)
                    # unchecked_id = torch.delete(unchecked_id, torch.where(unchecked_id==h_id))
                    unchecked_id = unchecked_id[torch.where(unchecked_id!=h_id)]

    # Set cells out side of field of view as unknown
    FOVmask = point_in_circle(x_local, y_local, ego_dict['pos'], FOV_radius, res) 
    sensor_grid[~FOVmask] = 0.5

    for id in unchecked_id:
        mask1 = (label_grid[1,:,:]==id)
        if torch.any(sensor_grid[mask1] == 1.):
            sensor_grid[mask1] = 1. 
            visible_id.append(id)

        else:
            pass
    
    ## plot and save the sensor_grid to see the grid
    plt.imshow(sensor_grid.cpu().numpy())
    plt.savefig('sensor_grid.png')

    return visible_id, sensor_grid



def point_in_circle(x_local, y_local, center, radius, res):
    mask = torch.sqrt((x_local-center[0])**2+(y_local-center[1])**2) < radius  
    return mask

# generate the y indeces along a line
def linefunction(velx,vely,indx,indy,x_range):
    m = (indy-vely)/(indx-velx)
    b = vely-m*velx
    return m*x_range + b 

def pointinpolygon(x,y,poly):
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


def parallelpointinpolygon(points, polygon):
    D = torch.empty(len(points), dtype=torch.bool) 
    for i in range(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D    





