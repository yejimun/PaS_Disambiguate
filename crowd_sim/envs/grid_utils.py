import numpy as np
from scipy import *
import copy
import torch
from scipy.spatial import cKDTree
from numba import jit, njit
from shapely.geometry import Point, LineString
import random
import numba
# from mppi.utils import *
import pdb


# TODO: There should be a better way to do this in Pytorch
# TODO: indice matching should be done in a batched manner
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


def Transfer_to_EgoGrid(curr_xy, curr_grids, next_xy, next_grids):
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
    
    # pred_maps_egoGrid = [] # pred_maps in ego grid

    curr_grids_ = curr_grids.clone()
    updated_next_grids = transfer_grid_data(curr_xy, curr_grids_, next_xy, next_grids ,distance_threshold=global_res)

    return updated_next_grids

# IS metric.
def MapSimilarityMetric(pas_map, sensor_grid, label_grid):    
    # Making a hard binary grids    
    gt_map = label_grid
    pas_map = np.where(pas_map>=0.6, 1., pas_map)
    pas_map = np.where(pas_map<=0.4, 0., pas_map)
    pas_map = np.where(np.logical_and(pas_map!=1., pas_map!=0.), 0.5, pas_map)

    A = pas_map.clone()
    B = gt_map.clone()
    base_A = sensor_grid.clone()
    
    psi_occupied, psi_free, psi_occluded = computeSimilarityMetric(B, A)
    base_psi_occupied, base_psi_free, base_psi_occluded = computeSimilarityMetric(B, base_A)
    psi_sum = psi_occupied + psi_free + psi_occluded
    base_psi_sum = base_psi_occupied + base_psi_free + base_psi_occluded
        
    psi = [psi_sum, psi_occupied, psi_free, psi_occluded ]
    base_psi = [base_psi_sum, base_psi_occupied, base_psi_free, base_psi_occluded]
        
    return psi, base_psi

def toDiscrete(m):
    """
    Args:
        - m (m,n) : np.array with the occupancy grid
    Returns:
        - discrete_m : thresholded m
    """
    m_occupied = np.zeros(m.shape)
    m_free = np.zeros(m.shape)
    m_occluded = np.zeros(m.shape)

    m_occupied[m == 1.0] = 1.0
    m_occluded[m == 0.5] = 1.0
    m_free[m == 0.0] = 1.0

    return m_occupied, m_free, m_occluded

def todMap(m):

    """
    Extra if statements are for edge cases.
    """

    y_size, x_size = m.shape
    dMap = np.ones(m.shape) * np.Inf
    dMap[m == 1] = 0.0

    for y in range(0,y_size):
        if y == 0:
            for x in range(1,x_size):
                h = dMap[y,x-1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(0,x_size):
                if x == 0:
                    h = dMap[y-1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y,x-1]+1, dMap[y-1,x]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    for y in range(y_size-1,-1,-1):

        if y == y_size-1:
            for x in range(x_size-2,-1,-1):
                h = dMap[y,x+1]+1
                dMap[y,x] = min(dMap[y,x], h)

        else:
            for x in range(x_size-1,-1,-1):
                if x == x_size-1:
                    h = dMap[y+1,x]+1
                    dMap[y,x] = min(dMap[y,x], h)
                else:
                    h = min(dMap[y+1,x]+1, dMap[y,x+1]+1)
                    dMap[y,x] = min(dMap[y,x], h)

    return dMap

def computeDistance(m1,m2):

    y_size, x_size = m1.shape
    dMap = todMap(m2)

    d = np.sum(dMap[m1 == 1])
    num_cells = np.sum(m1 == 1)

    # If either of the grids does not have a particular class,
    # set to x_size + y_size (proxy for infinity - worst case Manhattan distance).
    # If both of the grids do not have a class, set to zero.
    if ((num_cells != 0) and (np.sum(dMap == np.Inf) == 0)):
        output = d/num_cells
    elif ((num_cells == 0) and (np.sum(dMap == np.Inf) != 0)):
        output = 0.0
    elif ((num_cells == 0) or (np.sum(dMap == np.Inf) != 0)):
        output = x_size + y_size

    if output == np.Inf:
        pdb.set_trace()

    return output

def computeSimilarityMetric(m1, m2):

    m1_occupied, m1_free, m1_occluded = toDiscrete(m1)
    m2_occupied, m2_free, m2_occluded = toDiscrete(m2)

    occupied = computeDistance(m1_occupied,m2_occupied) + computeDistance(m2_occupied,m1_occupied)
    occluded = computeDistance(m2_occluded, m1_occluded) + computeDistance(m1_occluded, m2_occluded)
    free = computeDistance(m1_free,m2_free) + computeDistance(m2_free,m1_free)

    return occupied, free, occluded


def rotate_around_center(pts, center, yaw):
    return np.dot(pts - center, np.array([[np.cos(yaw), np.sin(yaw)], [-np.sin(yaw), np.cos(yaw)]])) + center


# Get state of a vehicle at timestamp when its id is known
def getstate(timestamp, track_dict, id):
    for key, value in track_dict.items():
        if key==id:
            return value.motion_states[timestamp]



# reshape list
def reshape(seq, rows, cols):
    return [list(u) for u in zip(*[iter(seq)] * cols)]
    
# helper function from pykitti
def rotx(t):
    """Rotation about the x-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])

# helper function from pykitti
def roty(t):
    """Rotation about the y-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])

# helper function from pykitti
def rotz(t):
    """Rotation about the z-axis."""
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],s
                     [s,  c,  0],
                     [0,  0,  1]])

# helper function from pykitti
def pose_from_oxts_packet(lat,lon,alt,roll,pitch,yaw,scale):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    tz = alt
    t = np.array([tx, ty, tz])

    # Use the Euler angles to get the rotation matrix
    Rx = rotx(roll)
    Ry = roty(pitch)
    Rz = rotz(yaw)
    R = Rz.dot(Ry.dot(Rx))

    # Combine the translation and rotation into a homogeneous transform
    return R, t

# position of vertex relative to global origin (adapted from pykitti)
def pose_from_GIS(lat,lon,scale,origin):
    """Helper method to compute a SE(3) pose matrix from an OXTS packet.d
    """
    er = 6378137.  # earth radius (approx.) in meters

    # Use a Mercator projection to get the translation vector
    tx = scale * lon * np.pi * er / 180.
    ty = scale * er * \
        np.log(np.tan((90. + lat) * np.pi / 360.))
    # 2D position
    t = np.array([tx, ty])

    return (t-origin[0:2])

# helper function from pykitti
def transform_from_rot_trans(R, t):
    """Homogeneous transformation matrix from rotation matrix and translation vector."""
    R = R.reshape(3, 3)
    t = t.reshape(3, 1)
    return np.vstack((np.hstack([R, t]), [0, 0, 0, 1]))

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) >= (B[1]-A[1]) * (C[0]-A[0])

# Return true if line segments AB and CD intersect
def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

@jit(nopython=True)
def pointinpolygon(x,y,poly):
    n = len(poly)
    inside = False
    p2x = 0.0
    p2y = 0.0
    xints = 0.0
    p1x,p1y = poly[0]
    for i in numba.prange(n+1):
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


@njit 
def parallelpointinpolygon(points, polygon):
    D = np.empty(len(points), dtype=numba.boolean) 
    for i in numba.prange(0, len(D)):
        D[i] = pointinpolygon(points[i,0], points[i,1], polygon)
    return D    



def point_in_rectangle(x, y, rectangle):
    A = np.array(rectangle[0])
    B = np.array(rectangle[1])
    C = np.array(rectangle[2])

    M = np.array([x,y]).transpose((1,2,0)) # (100,100,2)

    AB = B-A
    AM = M-A # nxnx2
    BC = C-B
    BM = M-B # nxnx2

    dotABAM = np.dot(AM,AB) # nxn
    dotABAB = np.dot(AB,AB)
    dotBCBM = np.dot(BM,BC) # nxn
    dotBCBC = np.dot(BC,BC)

    return np.logical_and(np.logical_and(np.logical_and((0. <= dotABAM), (dotABAM <= dotABAB)), (0. <= dotBCBM)), (dotBCBM <= dotBCBC)) # nxn

# create a grid in the form of a numpy array with coordinates representing
# the middle of the cell (30 m ahead, 30 m to each side, and 30 m behind)
# cell resolution: 0.33 cm

def global_grid(origin,endpoint,res):

    xmin = min(origin[0],endpoint[0]) 
    xmax = max(origin[0],endpoint[0]) + res/2.
    ymin = min(origin[1],endpoint[1]) 
    ymax = max(origin[1],endpoint[1]) + res/2. 

    x_coords = np.arange(xmin,xmax,res)
    y_coords = np.arange(ymin,ymax,res)

    gridx,gridy = np.meshgrid(x_coords,y_coords)

    return gridx, np.flipud(gridy) 

def point_in_circle(x_local, y_local, center, radius, res):
    mask = np.sqrt(np.power(x_local-center[0],2)+ np.power(y_local-center[1],2)) < radius  
    return mask


def find_nearest(n,v,v0,vn,res): 
    "Element in nd array closest to the scalar value `v`" 
    idx = int(np.floor( n*(v-v0+res/2.)/(vn-v0+res) ))
    return idx

# generate the y indeces along a line
def linefunction(velx,vely,indx,indy,x_range):
    m = (indy-vely)/(indx-velx)
    b = vely-m*velx
    return m*x_range + b 

def run_ransac(data, estimate, is_inlier, sample_size, goal_inliers, max_iterations, stop_at_goal=True, random_seed=None):
    best_ic = 0
    best_model = None
    random.seed(random_seed)
    for i in range(max_iterations):
        s = data[np.random.choice(data.shape[0], 3, replace=False), :]
        m = estimate(s)
        ic = 0
        for j in range(data.shape[0]):
            if is_inlier(m, data[j,:]):
                ic += 1
        if ic > best_ic:
            best_ic = ic
            best_model = m
            if ic > goal_inliers and stop_at_goal:
                break
    print ('took iterations:', i+1, 'best model:', best_model, 'explains:', best_ic)
    return best_model, best_ic

def augment(xyzs):
	axyz = np.ones((len(xyzs), 4))
	axyz[:, :3] = xyzs
	return axyz

def estimate(xyzs):
	axyz = augment(xyzs[:3])
	return np.linalg.svd(axyz)[-1][-1, :]

def is_inlier(coeffs, xyz, threshold):
	return np.abs(coeffs.dot(augment([xyz]).T)) < threshold

def is_close(a,b,c,d,point,distance=0.1):
    D = (a*point[:,0]+b*point[:,1]+c*point[:,2]+d)/np.sqrt(a**2+b**2+c**2)
    return D


############ For polygon ray tracing ##########################
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """        

    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    # If angles between two vectors are bigger than or equal to 180 degree
    # if np.cross(v1_u, v2_u)<0 or (np.cross(v1_u, v2_u)==0 and np.dot(v1_u, v2_u)<0) :
    #     print("here")
    #     return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)) + np.pi
    # else:
    #     print("else")
    angle = np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)
    if angle == -1.0:
        return -np.pi
    elif angle == 1.0:
        return np.pi
    else:
        return np.arccos(angle)

def clip_polygon_to_grid(poly_verts, grid_polygon):
    """
    Given poly_verts (an ordered list of [x, y] points)
    and a grid_polygon (a Shapely Polygon), return new vertices
    for the polygon clipped to the grid.
    """
    new_poly = []
    n = len(poly_verts)
    for i in range(n):
        p1 = Point(poly_verts[i])
        p2 = Point(poly_verts[(i + 1) % n])
        edge = LineString([poly_verts[i], poly_verts[(i + 1) % n]])

        # If p1 is inside, keep it.
        if grid_polygon.contains(p1):
            new_poly.append(poly_verts[i])
        else:
            # For vertices outside, we may add the intersection if the edge crosses the grid.
            # (We don't add p1 because it lies outside.)
            pass

        # Check if the edge between p1 and p2 intersects the grid boundary.
        intersection = edge.intersection(grid_polygon.boundary)
        if not intersection.is_empty:
            # The intersection can be a single point or multiple points.
            if intersection.geom_type == 'Point':
                inter_pt = [intersection.x, intersection.y]
                # Only add if it's not already the last point.
                if not new_poly or np.any(new_poly[-1] != inter_pt):
                    new_poly.append(inter_pt)
            elif intersection.geom_type == 'MultiPoint':
                for pt in intersection:
                    inter_pt = [pt.x, pt.y]
                    if not new_poly or np.any(new_poly[-1] != inter_pt):
                        new_poly.append(inter_pt)

    return new_poly