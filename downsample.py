import numpy as np
import open3d as o3d 
import time
import os 

def sanity_check(file_path):
    start = time.clock()
    point_cloud = o3d.io.read_point_cloud(file_path)

    print("has points:", point_cloud.has_points())
    print("has colors:", point_cloud.has_colors())
    print("loading time:", time.clock() - start, "s")

    o3d.visualization.draw_geometries([point_cloud])


def down_sample(file_path, voxel_size):
    point_cloud = o3d.io.read_point_cloud(file_path)

    return point_cloud.voxel_down_sample(voxel_size)

def down_sample_write(file_prefix, voxel_size):
    """ writes downsampled """ 
    pcd_file = file_prefix + ".pcd"
    pcd_file_downsampled = file_prefix + "_downsampled_" +  str(voxel_size) + ".pcd"

    if os.path.isfile(pcd_file_downsampled):
        print("pcd file already exists")
        return
    
    downsampled_pcd = down_sample(pcd_file, voxel_size)
    written = o3d.io.write_point_cloud(pcd_file_downsampled, downsampled_pcd)

    if written: 
        print("downsampled written correctly for voxel size:", voxel_size, "!")
    else: 
        print("downsampled failed")

if __name__ == "__main__":
    #sanity_check("data/36_N_First_RCS.pcd")
    down_sample_write("data/36_N_First_RCS", 0.1)
    print("voxel down sampled")
