import numpy as np
#import pandas as pd
from sklearn.decomposition import PCA
import open3d as o3d 

import preprocess

#use pca on all points within a radius of r_pca
r_pca = 0.0
t_planar1, t_planar2 = 1.8, 2.0
t_scatter1, t_scatter2 = 2.0 ,1.8
t_linear = 2.0

LINEAR = 2
PLANAR = 0
SCATTER = 1
OTHER = -1

def grow_region(pcd, region, r_seg, k_nn, search):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    seen = set()
    grown_region = o3d.utility.IntVector()
    
    for point in region:
        if point not in seen:            
            if search == 'radius':
                [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[point], r_seg)
            elif search == 'knn':
                [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[point], k_nn)
                      
            seen.add(point)      
            for j in idx:
                if j not in seen:
                    seen.add(j)
                
    grown_region = list(seen)
    return grown_region
def grouping(pcd, r_seg, k_nn, search, all_set):

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    seen = set()
    components = []
    for i in range(len(pcd.points)):
        if i in seen:
            continue
        if all_set[i] != SCATTER:
            seen.add(i)
            continue
            
        new_component = o3d.utility.IntVector()
    
        queue = [i]
        while len(queue) > 0:
            index = queue.pop(0)
            if index not in seen:
                seen.add(index)
                new_component = np.append(new_component, index)

                if search == 'radius':
                    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[index], r_seg)
                elif search == 'knn':
                    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[index], k_nn)
                
                if k < 3:
                    continue
                for j in idx:
                    if j not in seen and all_set[j] == SCATTER:
                        queue.append(j)

        components.append(new_component)

    final_components = []
    for c in components:
        if len(c) > 1000:
            final_components.append(c)
    return final_components
def find_facade(pcd, scatter, planar, linear, dist_thresh):
    """ uses open3d ransac algorithm to find a facade""" 
    
    _, inliers = pcd.segment_plane(distance_threshold=dist_thresh,
                                        ransac_n=3,
                                         num_iterations=1000)

    for index in inliers:
        if index in scatter:
            scatter.remove(index)
            planar.append(index)
        elif index in linear:
            linear.remove(index)
            planar.append(index)

def isolated_density_filter(pcd, r_threshhold):
    """ remove_radius_outliers removes every point which has fewer than 5 neighbors within a radius"""
    pcd.remove_radius_outlier(15, r_threshhold)

def segment_scatter_planar(pcd, r_pca=1.0, k_nn = 500, search='radius'):
    """ 
    takes in point cloud object, and determines scatter and planar points
    returns tuple of 4 open3d IntVectors
    """ 
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(pcd.points)

    scatter_set = o3d.utility.IntVector()
    linear_set = o3d.utility.IntVector()
    planar_set = o3d.utility.IntVector()
    other_set = o3d.utility.IntVector()
    all_set = o3d.utility.IntVector()

    for i in range(num_points):
        if search == 'radius':
            [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], r_pca)
        elif search == 'knn':
            [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], k_nn)
    
        if k < 3:
            continue

        ith_point_nearest_neighbors = np.asarray(pcd.points)[idx, :]
        #classify_ith = classify_eigenvals(calculate_pca_of_set(ith_point_nearest_neighbors))
        classify_ith = classify_p_values(convert_pca_to_p(calculate_pca_of_set(ith_point_nearest_neighbors)))
        if classify_ith == SCATTER:
            all_set.append(SCATTER)
            scatter_set.append(i)
        elif classify_ith == LINEAR:
            all_set.append(LINEAR)
            linear_set.append(i)
        elif classify_ith == PLANAR:
            all_set.append(PLANAR)
            planar_set.append(i)
        else:
            all_set.append(OTHER)
            other_set.append(i)
        
        
    return all_set, scatter_set, linear_set, planar_set, other_set

def color_pcd_points(pcd, scatter_set, linear_set, planar_set, other_set):
    #scatter
    colors = np.asarray(pcd.colors)
    colors[scatter_set] = [1, 0, 0]
    colors[linear_set] = [0, 0, 1]
    colors[planar_set] = [0, 1, 0]
    colors[other_set] = [0.5, 0.5, 0.5]
    
def calculate_pca_of_set(point_set):
    """
    Takes in a numpy array and calculates principle components of the set
    input: numpy array
    output: principle component eigenvalues
    """ 
    point_set_pca = PCA(n_components=3)
    point_set_pca.fit(point_set)
    return point_set_pca.singular_values_
    
def convert_pca_to_p(eigenvalues):
    """this function takes lambda_max, _mid, _min,
    and calculates _max / (_max + _mid + _ min) for each principle component
    potentially better measure of eigenvalue classification than pure ratio
    """ 
    lambda_max, lambda_mid, lambda_min = eigenvalues[0], eigenvalues[1], eigenvalues[2]
    total_sum = lambda_max + lambda_mid + lambda_min

    return [lambda_max / total_sum, lambda_mid / total_sum, lambda_min / total_sum]

def classify_p_values(p_vals):
    """Brodu and Lague"""

    p1, p2, p3 = p_vals[0], p_vals[1], p_vals[2]

    if p1 > 0.33 and p1 - p2 > 0.22:
        return LINEAR
    elif p1 - p2 < 0.15 and p3 < 0.25:
       return PLANAR
    elif p1 - p3 < 0.33: 
        return SCATTER
    else: 
        return OTHER
    
    return 

    
def classify_eigenvals(eigenvalues): #maybe replace this function with 
    """
    classify function takes in eigenvalue of point_set, returns label
    input: numpy array
    output: label (0-2)
    """
    lambda_max, lambda_mid, lambda_min = eigenvalues[0], eigenvalues[1], eigenvalues[2]
    
    if lambda_max / lambda_mid > t_linear:
        return LINEAR
    elif lambda_min == 0 or lambda_mid / lambda_min > t_planar1 and lambda_max / lambda_mid < t_planar2:
        return PLANAR
    elif lambda_max / lambda_mid < t_scatter1 and lambda_mid / lambda_min < t_scatter2:
        return SCATTER
    else:
        return OTHER

def classify_label(label):
    if label == PLANAR:
        return "Planar"
    elif label == SCATTER: 
        return "Scatter"
    else:
        return "Other"

if __name__ == "__main__":
    #data = preprocess.txt_to_pandas("data/36_N_First_RCS.txt")
    test = np.array([[1, 0, 0],[0, 1, 0], [-1, 0, 0], [0, -1, 0]])
    eigen_vals = calculate_pca_of_set(test)
    print(eigen_vals)
    label = classify_eigenvals(eigen_vals)
    print(classify_label(label))