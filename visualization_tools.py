import numpy as np
import open3d as o3d

def visualize_3d_equation(pcd, equation, distance_threshhold, color):
    """ takes in a point cloud and an equation, calculates which points satisfy the equation"""
    [a, b, c, d] = equation
    num_points = len(pcd.points)
    inliers = o3d.utility.IntVector()
    for i in range(num_points):
        x, y, z = pcd.points[i]
        abs_equation_value = np.abs(a * x + b * y + c * z + d)
        if abs_equation_value < distance_threshhold:
            inliers.append(i)
    
    print(f"Plane equation facade: {a:.2f}x + {b:.2f}y + {c:.2f}z + {d:.2f} = 0")

    np.asarray(pcd.colors)[inliers,:] = color
    o3d.visualization.draw_geometries([pcd])