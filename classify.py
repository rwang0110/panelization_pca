import numpy as np
#import pandas as pd
from sklearn.decomposition import PCA
import open3d as o3d 

import preprocess

#use pca on all points within a radius of r_pca
r_pca = 0.0
t_planar, t_scatter1, t_scatter2, t_linear = 2.3, 1.5, 1.65, 2.0
LINEAR = 2
PLANAR = 0
SCATTER = 1
OTHER = -1


def segment_scatter_planar(pcd, r_pca=1.0, search='radius'):
    """ 
    takes in point cloud object, and determines scatter and planar points 
    """ 
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(pcd.points)

    for i in range(num_points):
        if search == "radius":
            [_, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[i], r_pca)
        else:
            [_, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[i], r_pca)
        ith_point_nearest_neighbors = np.asarray(pcd.points)[idx[1:], :]
        classify_ith = classify_eigenvals(calculate_pca_of_set(ith_point_nearest_neighbors))
        if classify_ith == SCATTER:
            pcd.colors[i] = [1, 0, 0] 
        if classify_ith == LINEAR:
            pcd.colors[i] = [0, 0, 1]
    return

def calculate_pca_of_set(point_set):
    """
    Takes in a numpy array and calculates principle components of the set
    input: numpy array
    output: principle component eigenvalues
    """ 
    point_set_pca = PCA(n_components=3)
    point_set_pca.fit(point_set)
    return point_set_pca.singular_values_
    
    
def classify_eigenvals(eigenvalues): #maybe replace this function with 
    """
    classify function takes in eigenvalue of point_set, returns label
    input: numpy array
    output: label (0-2)
    """
    lambda_max, lambda_mid, lambda_min = eigenvalues[0], eigenvalues[1], eigenvalues[2]
    if lambda_max / lambda_mid > t_linear:
        return LINEAR
    if lambda_min == 0 or lambda_mid / lambda_min > t_planar:
        return PLANAR
    elif lambda_max / lambda_mid < t_scatter1 and lambda_mid / lambda_min < t_scatter2:
        return SCATTER
    
    else:
        return OTHER
def classify_label_gauss(eigen_vals):
    """
    returns the most probable label
    lalondr: 
    S = (S_scatter, S_linear, S_surface)
    pca eigenvalues = lambda_max, lambda_mid, lamba_min
    S_scatter = lambda_max
    S_linear = lambda_max - lambda_mid
    S_surface = lambda_mid - lambda_min
    """
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