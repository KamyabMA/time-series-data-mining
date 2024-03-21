"""
In this file, different variants of the PIP algorithm (as defined in [Fu2008]) are implemented.
"""

import numpy as np
from icecream import ic


def dist_ED(p3, p1, p2) -> float:
    """
    Calculates the sum of euclidean distances of (p3, p1) and (p3, p2).
    p1, p2, and p3 shoult have the following format: [time(int or float), price(float)].

    Args:
        p3 (np.ndarray with shape: (2,))
        p2 (np.ndarray with shape: (2,))
        p1 (np.ndarray with shape: (2,))
    Returns:
        float: sum of euclidean distances of (p3, p1) and (p3, p2)
    """
    return np.sqrt((p3[0] - p1[0])**2 + (p3[1] - p1[1])**2) + np.sqrt((p3[0] - p2[0])**2 + (p3[1] - p2[1])**2)


def dist_VD(p3, p1, p2) -> float:
    """
    Calculates the vertical distance, which is the vertical distance of p3 to the line that connects p1 and p2.
    p1, p2, and p3 shoult have the following format: [time(int or float), price(float)].

    Note: Definition and visualization of the vertical distance in availabe in [Fu2008].

    Args:
        p3 (np.ndarray with shape: (2,))
        p2 (np.ndarray with shape: (2,))
        p1 (np.ndarray with shape: (2,))
    Returns:
        float: vertical distance
    """
    return abs((p1[1] + (p2[1] - p1[1])*((p3[0] - p1[0])/(p2[0] - p1[0]))) - p3[1])


def calc_pips(ts, n: int, dist):
    """
    Returns the first n Perceptually Important Points (PIPs) of the the input time sereis ts calculated using the distance function dist.
    The returned PIPs are sorted by time (not by importance).

    Args:
        ts (np.ndarray with shape (len(ts),2) - [[time, price]])
        n (int > 2)
        dist (distance function e.g., dist_VD)
    Returns:
        np.ndarray with shape (n,2) - [[time, price]]: sorted PIPs by time
    """
    # Initializations:

    m = len(ts)

    # List for storing the indices of the PIPs in ts:
    pip_indices = [0, m-1] # Initializing it with the first two PIPs.

    # Array for storing the index of the left adjacent PIP of non-PIP points in ts (we use this to calculate the distance):
    left_pip_index = np.zeros(shape=(m,), dtype=np.int32)
    # Initializing left_pip_index = [-1, 0, 0, ..., 0, -1], 
    # because the first and the last elemnts are PIPs and we do not calculate distances for these points (so -1),
    # and in the first iteration all the non-PIP elements in between have the first PIP (index=0) as their left adjacent PIP:
    left_pip_index[0] = -1
    left_pip_index[m-1] = -1
    
    # Array for storing the index of the right adjacent PIP of non-PIP points in ts (we use this to calculate the distance):
    right_pip_index = np.ones(shape=(m,), dtype=np.int32)
    # Initializing right_pip_index = [-1, m-1, m-1, ..., m-1, -1] (same reasoning as in the previous step):
    right_pip_index = right_pip_index * (m-1)
    right_pip_index[0] = -1
    right_pip_index[m-1] = -1

    # Array for storing the distances of non-PIPs to their adjacent PIPs:
    dists_arr = np.zeros(shape=(m,))
    # Initializing dists_arr = [-1, 0, 0, ..., 0, -1], because we did not calculated any distances yet:
    dists_arr[0] = -1
    dists_arr[m-1] = -1

    # Array for storing for each index of ts, whether we need to recalculate (1) the distance or we can use the previously calculated (cached) distance (0):
    recalculate_dists_arr = np.ones(shape=(m,), dtype=np.int32) # Initializing with [1, 1, 1, ..., 1, 1].
    
    # *** For debugging and testing *** 
    # ic(pip_indices)
    # ic(left_pip_index)
    # ic(right_pip_index)
    # ic(dists_arr)
    # ic(recalculate_dists_arr)

    # PIP algorithm:

    for i in range(2, n):
        
        if i == 2:
        # First Iteration
            # Calculate distances of non-PIPs to their adjacent PIPs (which are all points except the fist two points):
            for j in range(m):
                if dists_arr[j] > -1:
                    dists_arr[j] = dist(ts[j], ts[left_pip_index[j]], ts[right_pip_index[j]])
            # Find the point with the max distance (the new PIP):
            max_index = np.argmax(dists_arr)
            index_new_pip = max_index
            # Inserting the point as a new found PIP in pip_indices:
            pip_indices.insert(1, index_new_pip)
        
        if i > 2:
            # Recalculate the distance for some distances:
            for j in range(m):
                if recalculate_dists_arr[j] == 1:
                    dists_arr[j] = dist(ts[j], ts[left_pip_index[j]], ts[right_pip_index[j]])
            # Find the point with the max distance (the new PIP):
            max_index = np.argmax(dists_arr)
            index_new_pip = max_index
            # Inserting the point as a new found PIP in pip_indices:
            pos = 0
            while ts[pip_indices[pos]][0] < ts[index_new_pip][0]:
                pos += 1
            pip_indices.insert(pos, index_new_pip)

        # *** For debugging and testing *** 
        # print("----------")
        # ic(pip_indices)
    
        # Updating the arrays left_pip_index and right_pip_index:
        recalculate_dists_arr = np.zeros(shape=(m,), dtype=np.int32) # In this step we specify which distance to chache and which distance to recalculate.
        dists_arr[index_new_pip] = -1
        left_pip_index[index_new_pip] = -1
        right_pip_index[index_new_pip] = -1
        
        index_left_pip = pip_indices[pip_indices.index(index_new_pip)-1]  # the PIP left to the newly found PIP
        index_right_pip = pip_indices[pip_indices.index(index_new_pip)+1] # the PIP right to the newly found PIP
        for i in range(m):
            if index_new_pip < i and i < index_right_pip:
                recalculate_dists_arr[i] = 1
                left_pip_index[i] = index_new_pip
            if index_left_pip < i and i < index_new_pip:
                recalculate_dists_arr[i] = 1
                right_pip_index[i] = index_new_pip
        
        # *** For debugging and testing *** 
        # print("----------")        
        # ic(recalculate_dists_arr)
        # ic(left_pip_index)
        # ic(right_pip_index)
    
    # Returning the calculated PIPs:
    pips = np.empty(shape=(n,2))
    for i in range(len(pip_indices)):
        pips[i] = ts[pip_indices[i]]
    return pips


def pip_ED(ts, n: int):
    """
    Returns the first n Perceptually Important Points (PIPs) of the the input time sereis ts calculated using the 'dist_ED' distance.
    The returned PIPs are sorted by time (not by importance).

    Args:
        ts (np.ndarray with shape (len(ts),2) - [[time, price]])
        n (int > 2)
    Returns:
        np.ndarray with shape (n,2) - [[time, price]]: sorted PIPs by time
    """
    return calc_pips(ts, n, dist_ED)


def pip_VD(ts, n: int):
    """
    Returns the first n Perceptually Important Points (PIPs) of the the input time sereis ts calculated using the 'dist_VD' distance.
    The returned PIPs are sorted by time (not by importance).

    Args:
        ts (np.ndarray with shape (len(ts),2) - [[time, price]])
        n (int > 2)
    Returns:
        np.ndarray with shape (n,2) - [[time, price]]: sorted PIPs by time
    """
    return calc_pips(ts, n, dist_VD)
