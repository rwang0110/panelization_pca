""" this document is meant to segmenet the data into sets of local neighborhoods""" 
import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors

import preprocess
import classify
def segment_data_nearest_neighbors(data_array):
    """ 
    segments data into neighbors based on nearest neighbors algorithm
    input: data_array, numpy array, r_pca, radius
    output: numpy array listing neighborhoods
    """
    neighbors = NearestNeighbors(n_neighbors=5, algorithm="ball_tree").fit(data_array)
    return neighbors.kneighbors(data_array)

def segment_data(data_array, r_pca, point_index):
    """ 
    takes in pandas file, returns local neighborhoods of points within sphere

    return Dataframe with points within 
    """ 
    res = np.array([data_array[point_index]])
    r_pca_sq = r_pca * r_pca
    print("distance:", r_pca_sq)
    for i in range(len(data_array)): 
        if i == point_index:
            continue
        if squared_distance_np(data_array, point_index, i) < r_pca_sq:
            res = np.vstack([res,data_array[i]])
    return res
def squared_distance_np(data_array, x_index, y_index):
    """ 
    takes in numpy array datafile, and indices of two points, calculates
    the distance between them
    """
    #print(x_point)
    res = 0.0
    for i in range(3):
        res += (data_array[x_index][i] - data_array[y_index][i]) * (data_array[x_index][i] - data_array[y_index][i])
    return res         
        
def squared_distance_pd(datafile, x_index, y_index):
    """ 
    takes in pandas datafile, and indices of two points, calculates
    the distance between them
    """
    #print(x_point)
    res = 0.0
    for i in range(3):
        res += (datafile.iloc[x_index][i] - datafile.iloc[y_index][i]) * (datafile.iloc[x_index][i] - datafile.iloc[y_index][i])
    return res

if __name__ == "__main__":
    start_time = time.clock()
    column_names, data = preprocess.txt_to_pandas("data/36_N_First_RCS.txt")
    print("preprocess runtime:", time.clock() - start_time, "s")
    print(data)
    data = preprocess.reduce_to_xyz(data, column_names)
    print(data)
    start_time = time.clock()
    data = data.to_numpy()
    print("to numpy runtime: ", time.clock() - start_time, "s")
    
    #print("square distance: ", sq_dist)
    dataset = data
    print(dataset)

    #test runtime segmentation
    start_time = time.clock()
    distances, indices = segment_data_nearest_neighbors(dataset)
    print("segmentation runtime: ", time.clock() - start_time, "s\n")
    print(distances.shape, indices.shape)
    print("distances:", distances[:100])
    print("indices:", indices[:100])