import numpy as np
import numba as nb
import pandas as pd 
from astropy import constants, units
from scipy.spatial import Delaunay
import scipy.interpolate as interp
import numpy.typing as npt
from typing import Tuple

######################################################################################################################################################

# Ray tracer stuff

######################################################################################################################################################

def norm(arr:np.ndarray, axis: int = None) -> float:
    if axis is None:
        total = 0.0
        for i in range(arr.shape[0]):
            total += arr[i] ** 2
        return np.sqrt(total)
    else:
        output = np.empty(arr.shape[0])
        for i in range(arr.shape[0]):
            total = 0.0
            for j in range(arr.shape[1]):
                total += arr[i, j] ** 2
            output[i] = np.sqrt(total)
        return output

@nb.njit
def start_point(position: np.ndarray, r0: float) -> int:
    """
    Find the index of the closest point to the starting point using squared distances.
    """
    min_dist = np.inf  # Initialize with a very large value
    min_index = -1     # Placeholder for the index

    for i in range(position.shape[0]):
        dist = 0.0
        for j in range(position.shape[1]):
            diff = position[i, j] - r0[j]  # Element-wise difference
            dist += diff * diff            # Accumulate squared distance

        if dist < min_dist:
            min_dist = dist
            min_index = i

    return min_index


def get_distance_point_to_ray(start_point:float, unit_vector:np.ndarray , point:float) -> float:
    """   
    Given a starting point and a unit vector (i.e. a ray), and a point, this function returns the distance between the point and the ray.
    """
    return np.linalg.norm(np.cross(point - start_point, unit_vector))/np.linalg.norm(unit_vector)

def get_next_point_on_ray(start_point:npt.ArrayLike, unit_vector:npt.ArrayLike, points: npt.ArrayLike) -> npt.ArrayLike:
    """   
    Given a starting point, a unit vector and an array of points (neighbours), this function returns the points projected onto the ray.
    """
    vectors = points - start_point
    projections = np.dot(vectors, unit_vector)[:, np.newaxis] * unit_vector
    return start_point + projections

def get_next(start_point:np.ndarray, index:int, neighbors:np.ndarray, position:np.ndarray, unit_vector:np.ndarray, global_start_point:np.ndarray) -> Tuple[np.ndarray, int]:
    """  
    From a certain point, this function returns the next point on the ray and its index.
    """
    # Get the neighbors of the current point
    neighbors_current_point_ind = neighbors[index]
    neighbors_current_point_pos = position[neighbors_current_point_ind]

    # Calculate new points on the ray by projecting the neighbors onto the ray
    new_points_on_ray = get_next_point_on_ray(start_point, unit_vector, neighbors_current_point_pos)
    
    # Calculates the perpendicular distances to the ray of all neighbouring points
    distances_to_ray = np.linalg.norm(np.cross(neighbors_current_point_pos - start_point, unit_vector) / np.linalg.norm(unit_vector), axis=1)

    # Calculates the distances to the global starting point to find which points lie in the direction of the ray
    distances_to_start = np.linalg.norm(new_points_on_ray - global_start_point, axis=1)
    distance_to_start_previous = np.linalg.norm(start_point - global_start_point)
        
    # Consider all points that lie further on the ray than the starting point
    mask = distance_to_start_previous < distances_to_start 
    
    if np.sum(mask) == 0:
        # No valid point found, return an empty point and an invalid index (-1)
        return np.array([0.0, 0.0, 0.0]), np.int32(-1)
    
    # Take the neighboring points, in the mask, that lay closest to the ray, and select the one whose perpendicular distance to the ray is the smallest
    best_neighbor = np.argmin(distances_to_ray[mask])
    valid_neighbors = neighbors_current_point_ind[mask]  # Filtered neighbors based on mask
    next_index = valid_neighbors[best_neighbor]
    
    # The next point is the point that lies closest to the ray and is further on the ray than the starting point
    next_point = new_points_on_ray[mask][best_neighbor]
    
    return next_point, next_index

def get_all_points(start_point_ray:np.ndarray, direction:np.ndarray, neighbors:np.ndarray, position:np.ndarray, boundary:np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """  
    Calculates all the points on a ray in a certain direction, and the indices of the projected points
    """
    points, indices = np.zeros((500, 3)), np.zeros(500, dtype=np.int32)
    
    index0 = start_point(position, start_point_ray)
    points[0], indices[0] = start_point_ray, index0
    unit_vector = direction/np.linalg.norm(direction)
    
    # Just define some point far away to find the points in a certain direction
    some_point = start_point_ray - 100*unit_vector*np.linalg.norm(position[neighbors[index0][0]])
    
    next_index, next_point = index0, start_point_ray
    next_point, next_index = get_next(next_point, next_index, neighbors, position, unit_vector, some_point)
    
    if next_index == -1:
        print(" WARNING: Ray is too short")
        return points, indices
    
    points [1] = next_point
    indices[1] = next_index
        
    c = 1
        
    #while not check_boundary(next_index, boundary, index0):
    while next_index not in boundary:    
        
        c += 1 
    
        next_point, next_index = get_next(next_point, next_index, neighbors, position, unit_vector, some_point)
        
        if next_index == -1:
            return points, indices
        
        points [c] = next_point
        indices[c] = next_index
            
    return points[:c], indices[:c]
    

#####################################################################################################################



#####################################################################################################################

def get_ds(ray:np.ndarray) -> np.ndarray:
    '''
    Calculates the distance increment along subsequent points of a ray in cm
    '''
    ds = 100 * norm(ray[1:] - ray[:-1], axis=1)  # times 100 to convert to cm (orignally in m)
    return ds

def get_column_density(ray: np.ndarray, indices:np.ndarray, nH2:np.ndarray) -> float:
    """    
    Calculates the column density along a ray in [1/cm^2]
    """
    ds = get_ds(ray)
       
    column_density = 0
    n = nH2[indices]
    
    # Because of the way rays are traced, you need to stop when it encounters the boundary by hand...
    nn = np.where(indices == 0)[0][0]
    
    if len(indices) <= 1:
        return 0

    for i in range(0, nn-1):
        column_density += 0.5 * (n[i] + n[i + 1]) * ds[i]
        
    return column_density


######################################################################################################################################################

# Iteration stuff

######################################################################################################################################################

def get_rel_difference_in_x(x1, x2):
    diff = np.abs(x1 - x2) / x1
    diff[ np.isnan(diff) ] = 0
    return diff

@nb.njit
def custom_interpolate(r:np.ndarray, x:np.ndarray, r_to_interp:float) -> float:
    n = len(r)
    
    if r_to_interp < r[0]:
        return x[0]
    for j in range(n - 1):
        if r[j] <= r_to_interp <= r[j + 1]:
            r0, r1 = r[j], r[j + 1]
            x0, x1 = x[j], x[j + 1]
        
            return x0 + (r_to_interp - r0) * (x1 - x0) / (r1 - r0)
    return x[-1]

@nb.njit
def do_interpolation(r_r_new, closest_rays_indices, a_xs, a_r):
    
    n = len(r_r_new)
    m = len(closest_rays_indices[0])
    
    res = np.zeros((n, m))
    
    for i in range(n):
        for j in range(m):
            
            x = a_xs[closest_rays_indices[i][j]]
            r = a_r[closest_rays_indices[i][j]]
            r_interp = r_r_new[i][j]
            
            res[i][j] = custom_interpolate(r, x, r_interp)
    return res

@nb.njit
def calculate_x(dists:np.ndarray, xs:np.ndarray, k:float) -> float:
    return ( 1 / np.sum(1 / (dists**k)) ) * np.sum( xs / (dists**k) )

@nb.njit
def do_calculate_x(res, new_distances, k):
    
    n = len(res)
    x_res = np.zeros(n)
    
    for i in range(n):
        x_res[i] = calculate_x(new_distances[i], res[i], k)
        
    return x_res