import numpy as np
import pandas as pd
import time
from sklearn.neighbors import NearestNeighbors

import open3d as o3d 
import preprocess
import classify
import vectorfunctions

def compute_equation_by_groups(pcd, groups, dist_thresh):
    """ given a list of inliers in the pcd, returnt the equation and inliers of the equation """
    equations = []
    for surface in groups:
        surface_pcd = pcd.select_by_index(surface)

        eq, _ = surface_pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
        equations.append(eq)
        
    return equations

def group_by_normals(pcd, r_seg, k_nn, search):
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    num_points = len(pcd.points)
    seen = set()
    components = []
    for i in range(num_points):
        if i in seen:
            continue
        if i % 1000 == 0:
            print("point", i, "of", num_points)
        new_component = o3d.utility.IntVector()
    
        queue = [i]
        while len(queue) > 0:
            index = queue.pop(0)
            if index not in seen:
                seen.add(index)
                new_component = np.append(new_component, index)

                if search == 'radius'  :
                    [k, idx, _] = pcd_tree.search_radius_vector_3d(pcd.points[index], r_seg)
                elif search == 'knn':
                    #radius search better
                    [k, idx, _] = pcd_tree.search_knn_vector_3d(pcd.points[index], k_nn)
                
                if k < 3:
                    continue
                for j in idx:
                    v1, v2 = pcd.normals[i], pcd.normals[j]
                    if j not in seen and vectorfunctions.compute_cos(v1, v2) > 0.9:
                        queue.append(j)
                     if j % 1000 == 0:
                        print("point", i, "of", num_points)
        components.append(new_component)

    final_components = []
    for c in components:
        if len(c) > 1000:
            final_components.append(c)
    return final_components
    
def grouping(pcd, r_seg, k_nn, search, all_set, label):

    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    seen = set()
    components = []
    for i in range(len(pcd.points)):
        if i in seen:
            continue
        if all_set[i] != label:
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
                    if j not in seen and all_set[j] == label:
                        queue.append(j)

        components.append(new_component)

    final_components = []
    for c in components:
        if len(c) > 1000:
            final_components.append(c)
    return final_components


def find_facades_ransac(pcd, num_facades, dist_thresh):
    """ 
    segments facades using ransac
    returns a set of groups
    """
    groups = []
    remaining_pcd = pcd
    for i in range(num_facades):
        print("iteration:", i + 1)
        
        equation, inliers = remaining_pcd.segment_plane(distance_threshold=dist_thresh, ransac_n=3, num_iterations=1000)
        [a, b, c, d] = equation
        print(f"Plane equation facade {i + 1:.2f}: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")
        groups.append([equation, inliers])
        remaining_pcd = remaining_pcd.select_by_index(inliers, invert=True)

    return groups


if __name__ == "__main__":
    print("hello world")