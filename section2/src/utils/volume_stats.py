"""
Contains various functions for computing statistics over 3D volumes
"""
import numpy as np

def Dice3d(a, b):
    """
    This will compute the Dice Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks -
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Dice3D. If you completed exercises in the lessons
    # you should already have it.
    # 1. Binary masks: True where hippocampus (class 1 or 2), False where background
    a_bin = a > 0
    b_bin = b > 0

    # 2. Count voxels
    overlap = np.sum(a_bin & b_bin)          # & = "in BOTH masks" -> the intersection
    total = np.sum(a_bin) + np.sum(b_bin)    # size of A + size of B

    # 3. Edge case: both masks empty -> nothing to find, and nothing found = perfect agreement
    if total == 0:
        return 1.0

    # 4. Dice = 2 x overlap / (A + B)
    return float(2.0 * overlap / total)

def Jaccard3d(a, b):
    """
    This will compute the Jaccard Similarity coefficient for two 3-dimensional volumes
    Volumes are expected to be of the same size. We are expecting binary masks - 
    0's are treated as background and anything else is counted as data

    Arguments:
        a {Numpy array} -- 3D array with first volume
        b {Numpy array} -- 3D array with second volume

    Returns:
        float
    """
    if len(a.shape) != 3 or len(b.shape) != 3:
        raise Exception(f"Expecting 3 dimensional inputs, got {a.shape} and {b.shape}")

    if a.shape != b.shape:
        raise Exception(f"Expecting inputs of the same shape, got {a.shape} and {b.shape}")

    # TASK: Write implementation of Jaccard similarity coefficient. Please do not use 
    # the Dice3D function from above to do the computation ;)
    # 1. Binary masks (same as Dice)
    a_bin = a > 0
    b_bin = b > 0

    # 2. Count voxels
    intersection = np.sum(a_bin & b_bin)     # in BOTH masks
    union = np.sum(a_bin | b_bin)            # | = "in AT LEAST ONE mask" -> the union

    # 3. Edge case: both empty -> perfect agreement
    if union == 0:
        return 1.0

    # 4. Jaccard = intersection / union
    return float(intersection / union)