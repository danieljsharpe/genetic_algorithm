'''
Python script containing functions to perform bucket sort on a numpy array A with A = {x : 0. <= x <= 1.}
Keeps track of the original indices and returns a list of sorted indices
'''

import numpy as np

# auxiliary function; insertion sort
def isort(A):

    if np.shape(A) <= 1: return A
    i = 1
    while i < np.shape(A)[0]:
        k = A[i]
        j = i - 1
        while j >= 0 and A[j] > k:
            A[j+1] = A[j]
            A[j] = k
            j -= 1
        i += 1
    return A

# main function; bucket sort
def bsort(A, nbuck=10):

    buckets = [[] for x in range(nbuck)]
    for i, x in np.ndenumerate(A):
        buckets[int(x*nbuck)].append((x, i[0]))
    out = []
    for bucket in buckets:
        try:
            out += [i[1] for i in isort(bucket)]
        except IndexError:
            out += isort(bucket)
    return out
