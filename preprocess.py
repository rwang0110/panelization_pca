import numpy as np
import pandas as pd
import open3d as o3d 
import os

def txt_to_pcd(file_prefix):
    """ converts text file to pcd"""
    pcd_file = file_prefix + ".pcd"

    if os.path.isfile:
        print("pcd {} exists, skipped".format(pcd_file))
        return

def txt_to_pandas(file_path): 
    """converts txt files to pandas dataframe"""
    
    print("processing...")
    file_dataframe = pd.read_csv(file_path, sep=" ", header=0)
    column_names = file_dataframe.columns
    print("processing complete!")
    return column_names, file_dataframe

def txt_to_numpy(file_path):
    
    file_array = np.loadtxt(file_path)
    return file_array

def reduce_to_xyz(dataframe, column_names):
    """ 
    drop the non-xyz columns of data
    """
    return dataframe.drop(columns=column_names[3:])

def segment_data_groups(dataframe):
    return
    
if __name__ == "__main__":
    column_names, data = txt_to_pandas("data/36_N_First_RCS.txt")
    print("head:", data.head())
    print("tail: ", data.tail())
    print("head:", data.head())
    print("tail: ", data.tail())
    print(data.to_numpy())
    print(data.shape)
    #format: xyz rgb intensity?