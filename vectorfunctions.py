import numpy as np


def magnitude(v1):
    """ takes in a numpy array, and computes the magnitude """
    return np.sqrt(np.dot(v1, v1))
    
def compute_cos(v1, v2):
    """ computes the square of the cosine of two vectors according to:
        cos^2 = (u * v )^2 / (u)^2 (v)^2
     """ 
    if v1.shape != v2.shape:
        return -1
    return  np.dot(v1, v2)/ (magnitude(v1) * magnitude(v2))
if __name__ == "__main__":
    #data = preprocess.txt_to_pandas("data/36_N_First_RCS.txt")
    a = np.array([1, 0, 0])
    b = np.array([1, 0, 1])

    print(compute_cos(a, b))
