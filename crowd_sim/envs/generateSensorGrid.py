import numpy as np
import math
from scipy import *
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from crowd_sim.envs.grid_utils import *
import warnings
warnings.filterwarnings('ignore')


############## Sensor_grid ##################
# 0:  empty
# 1 : occupied
# 0.5 :  unknown/occluded
#############################################


# Find the unknown cells using polygons for faster computation. (Result similar to ray tracing)
def generateSensorGrid(label_grid, ego_dict, ref_dict, wall_polygons, map_xy, FOV_radius, res=0.1, partial_visibility=True):
	x_local, y_local = map_xy
	
	center_ego = ego_dict['pos']
	occluded_id = []
	visible_id = []

	# get the maximum and minimum x and y values in the local grids
	x_shape = x_local.shape[0]
	y_shape = x_local.shape[1]	

	id_grid = label_grid[1].copy()


	unique_id = np.unique(id_grid) # does not include ego (robot) id
   
	# cells not occupied by ego itself
	mask = np.where(label_grid[0]!=2, True,False)

	# no need to do ray tracing if no object on the grid
	if np.all(label_grid[0,mask]==0.):
		sensor_grid = np.zeros((x_shape, y_shape))
	
	else:
		sensor_grid = np.zeros((x_shape, y_shape)) 

		ref_pos = np.array(ref_dict['pos'])
		ref_r = np.array(ref_dict['r'])

		# Find the cells that are occluded by the obstructing human agents
		# reorder humans according to their distance from the robot.
		distance = [np.linalg.norm(center-center_ego) for center in ref_pos]
		sort_indx = np.argsort(distance)
		unchecked_id = np.array(ref_dict['id'])[sort_indx]
		# Create occlusion polygons starting from closest humans. Reject humans that are already inside the polygons.
		for center, human_radius, h_id in zip(ref_pos[sort_indx], ref_r[sort_indx], unchecked_id):	
			# if human is already occluded, then just pass
			if h_id in occluded_id:
				continue

			hmask = (label_grid[1,:,:]==h_id)
			sensor_grid[hmask] = 1.

			alpha = math.atan2(center[1]-center_ego[1], center[0]-center_ego[0])
			theta = math.asin(np.clip(human_radius/np.sqrt((center[1]-center_ego[1])**2 + (center[0]-center_ego[0])**2), -1., 1.))
			
			# 4 or 5 polygon points
			# 2 points from human
			x1 = center_ego[0] + human_radius/np.tan(theta)*np.cos(alpha-theta)
			y1 = center_ego[1] + human_radius/np.tan(theta)*np.sin(alpha-theta)

			x2 = center_ego[0] + human_radius/np.tan(theta)*np.cos(alpha+theta)
			y2 = center_ego[1] + human_radius/np.tan(theta)*np.sin(alpha+theta)

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

			polygon_points = np.array([[x1, y1], [x2, y2], [x4, y4],[x3, y3]])
			grid_points = np.array([x_local.flatten(), y_local.flatten()])	


			occ_mask = parallelpointinpolygon(grid_points.T, polygon_points)
			occ_mask = occ_mask.reshape(x_local.shape)
			sensor_grid[occ_mask] = 0.5

			# check if any agent is fully inside the polygon
			for oid in unchecked_id:
				oid_mask = (label_grid[1,:,:]==oid)
				# if any agent is fully inside the polygon store in the occluded_id and opt from unchecked_id			
				if np.all(sensor_grid[oid_mask] == 0.5):
					occluded_id.append(oid)
					unchecked_id = np.delete(unchecked_id, np.where(unchecked_id==h_id)) # TODO: h_id -> oid??
     		
	if len(wall_polygons) > 0:
		# sensor grid with building
		sensor_grid = wall_raytracing(map_xy, sensor_grid, center_ego, wall_polygons)

	# Set cells out side of field of view as unknown
	FOVmask = point_in_circle(x_local, y_local, ego_dict['pos'], FOV_radius, res) 
	sensor_grid[np.invert(FOVmask)] = 0.5
 
	# Assuming wall is always visible
	for wall in wall_polygons:
		wall_mask = (label_grid[1,:,:]==-9999)
		sensor_grid[wall_mask] = 1.  # occupied / visible

	# sensor_grid = np.where(sensor_grid==2,0,sensor_grid)
	for id in unique_id:
		mask1 =	(label_grid[1,:,:]==id)
		if np.any(sensor_grid[mask1] == 1.):
			if not partial_visibility:
				sensor_grid[mask1] = 1.  # occupied / visible
			visible_id.append(id)	

	return visible_id, sensor_grid 


def wall_raytracing(local_xy, sensor_grid, center_ego, poly_verts):
	# check if the vertice is inside the grid.
	x_local, y_local = local_xy
	p1 = [x_local[0,0], y_local[0,0]]
	p2 = [x_local[0,-1], y_local[0,-1]]
	p3 = [x_local[-1,-1], y_local[-1,-1]]
	p4 = [x_local[-1,0], y_local[-1,0]]
	grid_vert = np.array([p1, p2, p3, p4])
 
	grid_polygon = Polygon(grid_vert)
	poly_of_interest = []
	for poly_vert in poly_verts:	
		# for vert in poly_vert.tolist():
		# 	print(vert, grid_polygon.contains(Point(vert)))
		for vert in poly_vert.tolist():
			if grid_polygon.contains(Point(vert)):
				poly_of_interest.append(poly_vert)
				break
	# print(grid_vert)
	# print(valid_vert)
	# if len(valid_vert)>0: 
	# get the maximum and minimum x and y values in the local grids
	x_min = np.amin(x_local)
	x_max = np.amax(x_local)
	y_min = np.amin(y_local)
	y_max = np.amax(y_local)
	grid_polygon = Polygon(grid_vert)

	# generate a line from the center indices to a the edge of the local grid: sqrt(2)*128./3. meters away (LiDAR distance)
	center_point = Point(center_ego[0], center_ego[1])
	
	for i, poly_vert in enumerate(poly_verts):	
		# if any of the polygon vertices is inside the grid, do the ray tracing, otherwise skip.
		start_raytracing = False
		for vert in poly_vert.tolist(): 
			if grid_polygon.contains(Point(vert)):
				start_raytracing = True
				# found the polygon points of interest inside the grid.
				polygon = Polygon(poly_vert)
				critical_vert = clip_polygon_to_grid(poly_vert, grid_polygon)
				critical_vert = np.array(poly_vert)
				break
		if start_raytracing:
			# check if center_ego is inside the polygon
			# if len(critical_vert) == 1: # ! To do the polygon ray tracing critical_vert should contain at list three points
			# 	critical_vert.extend([poly_vert[i-1], poly_vert[i+1]])
			# polygon = Polygon(poly_vert)
			if polygon.contains(center_point):
				continue
			if center_ego in poly_vert:
				np.delete(poly_verts, center_ego)
			# 1. Calculate the angles between all the polygon object vertices and the ego center in the local grid
			base_vec = np.mean(critical_vert,axis=0)-center_ego # heading vector  
			obj_vert_angles = [angle_between(base_vec, vert-center_ego) for vert in critical_vert]
			# print('base_point/angles',poly_vert[0], obj_vert_angles)
			# 2. Get the vertices with the biggest and smallest angle : (x1,y1), (x2, y2)
			[x1,y1] = list(critical_vert[np.argmin(obj_vert_angles)]) # 
			[x2,y2] = list(critical_vert[np.argmax(obj_vert_angles)]) # 
			# print('largest/smallest angle vertices', x1,y1, '/', x2,y2)
			
			# if center_ego[1] >= max(y1,y2) and x1>0 and center_ego[1]-max(y1,y2)<0.5:
			# 	print('center_ego', center_ego)
			# 	print('polygon points', critical_vert)
			# 	print('polygon points', obj_vert_angles)
			# 	pdb.set_trace()

			# points from the grids
			if x1 == center_ego[0]:
				x3 = center_ego[0]
				y3 = np.sign(y1-center_ego[1])*y_max
			elif x1 < center_ego[0]:
				x3 = min(x_min, 0) # ! Need to be bigger than 6
				y3 = linefunction(center_ego[0],center_ego[1], x1,y1,x3)
			else:
				x3 = max(x_max, 25)
				y3 = linefunction(center_ego[0],center_ego[1], x1,y1,x3)

			if x2 == center_ego[0]:
				x4 = center_ego[0]
				y4 = np.sign(y2-center_ego[1])*y_max
			elif x2 <= center_ego[0]:
				x4 = min(x_min, 0)
				y4 = linefunction(center_ego[0],center_ego[1],x2,y2,x4)
			else:
				x4 = max(x_max, 25)
				y4 = linefunction(center_ego[0],center_ego[1],x2,y2,x4)

			polygon_points = [[x1, y1], [x2, y2], [x4, y4],[x3, y3]]
				
			#  Check if additional grid vertices is needed.
		
			# 1. Define two polygon base vectors (no need to normalize)      
			base_vec1 = np.array([x3-x1, y3-y1])
			base_vec2 = np.array([x4-x2, y4-y2])

			if np.cross(base_vec1, base_vec2)>0:
				# ! For the right thumb's rule in 3.
				base_vec1 = np.array([x4-x2, y4-y2])
				base_vec2 = np.array([x3-x1, y3-y1])		    
			
			for vert in grid_vert:
				vec = vert - np.array(center_ego) 
				# 3. Check if any vertices are within the two polygon vectors
				if np.cross(vec, base_vec1) > 0 and np.cross(base_vec2, vec)> 0:
					polygon_points.insert(-1, vert)
     
			# if center_ego[1] > y1 and x1<0:
			# 	print('polygon points', polygon_points)
			# 	pdb.set_trace()

			polygon_points = np.array(polygon_points)
			grid_points = np.array([x_local.flatten(), y_local.flatten()])	

			# See the 
			occ_mask = parallelpointinpolygon(grid_points.T, polygon_points)
			occ_mask = occ_mask.reshape(x_local.shape)
			sensor_grid[occ_mask] = 0.5
	
			# # To plot the poly_verts for sanity check
			# occ_mask2 = parallelpointinpolygon(grid_points.T, poly_vert)
			# occ_mask2 = occ_mask2.reshape(x_local.shape)
			# sensor_grid[occ_mask2] = 2.
	return sensor_grid