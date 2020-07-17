import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import preprocess

#use pca on all points within a radius of r_pca
r_pca = 0.0
t_planar, t_scatter1, t_scatter2 = 2.0, 2.0, 2.0
PLANAR = 0
SCATTER = 1
OTHER = -1

def classify(sets):
    """ 
    takes in sets of points, calculates pca for each set
    input: array of sets of numpy array vectors
    output: attach labels to each point in each set
    """
    for s in sets:
        label = classify_eigenvals(calculate_pca_of_set(s))
    


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