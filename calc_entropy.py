# -*- coding: utf-8 -*-
"""
Created on Sat Oct 07 18:41:59 2017

@author: Abishek
"""

from __future__ import unicode_literals
import numpy as np
import itertools


def calc_entropy(time_series):
    m,n,o=np.shape(time_series)  #segments x samples x channels
    x=np.zeros(o)
    y=np.zeros(o)
    fin=[]
    for j in range(0,m):
        for i in range(0,o):        
            x[i]=shannon_entropy(time_series[j,:,i])
            y[i]=permutation_entropy(time_series[j,:,i],3,1)        
        z=np.concatenate((x, y))
    fin.append(z)
    
    fin = np.asarray(fin)

    return fin



def util_pattern_space(time_series, lag, dim):
    """Create a set of sequences with given lag and dimension

    Args:
       time_series: Vector or string of the sample data
       lag: Lag between beginning of sequences
       dim: Dimension (number of patterns)

    Returns:
        2D array of vectors

    """
    n = len(time_series)

    if lag * dim > n:
        raise Exception('Result matrix exceeded size limit, try to change lag or dim.')
    elif lag < 1:
        raise Exception('Lag should be greater or equal to 1.')

    pattern_space = np.empty((n - lag * (dim - 1), dim))
    for i in range(n - lag * (dim - 1)):
        for j in range(dim):
            pattern_space[i][j] = time_series[i + j * lag]

    return pattern_space


def util_standardize_signal(time_series):
    return (time_series - np.mean(time_series)) / np.std(time_series)


def util_granulate_time_series(time_series, scale):
    """Extract coarse-grained time series

    Args:
        time_series: Time series
        scale: Scale factor

    Returns:
        Vector of coarse-grained time series with given scale factor
    """
    n = len(time_series)
    b = int(np.fix(n / scale))
    cts = [0] * b
    for i in range(b):
        cts[i] = np.mean(time_series[i * scale: (i + 1) * scale])
    return cts



def permutation_entropy(time_series, m, delay):
    """Calculate the Permutation Entropy

    Args:
        time_series: Time series for analysis
        m: Order of permutation entropy
        delay: Time delay

    Returns:
        Vector containing Permutation Entropy

    Reference:
        [1] Massimiliano Zanin et al. Permutation Entropy and Its Main Biomedical and Econophysics Applications:
            A Review. http://www.mdpi.com/1099-4300/14/8/1553/pdf
        [2] Christoph Bandt and Bernd Pompe. Permutation entropy — a natural complexity
            measure for time series. http://stubber.math-inf.uni-greifswald.de/pub/full/prep/2001/11.pdf
        [3] http://www.mathworks.com/matlabcentral/fileexchange/37289-permutation-entropy/content/pec.m
    """
    n = len(time_series)
    permutations = np.array(list(itertools.permutations(range(m))))
    c = [0] * len(permutations)

    for i in range(n - delay * (m - 1)):
        # sorted_time_series =    np.sort(time_series[i:i+delay*m:delay], kind='quicksort')
        sorted_index_array = np.array(np.argsort(time_series[i:i + delay * m:delay], kind='quicksort'))
        for j in range(len(permutations)):
            if abs(permutations[j] - sorted_index_array).any() == 0:
                c[j] += 1

    c = [element for element in c if element != 0]
    p = np.divide(np.array(c), float(sum(c)))
    pe = -sum(p * np.log(p))
    return pe

def shannon_entropy(time_series):
    """Return the Shannon Entropy of the sample data.

    Args:
        time_series: Vector or string of the sample data

    Returns:
        The Shannon Entropy as float value
    """

    # Check if string
    if not isinstance(time_series, str):
        time_series = list(time_series)

    # Create a frequency data
    data_set = list(set(time_series))
    freq_list = []
    for entry in data_set:
        counter = 0.
        for i in time_series:
            if i == entry:
                counter += 1
        freq_list.append(float(counter) / len(time_series))

    # Shannon entropy
    ent = 0.0
    for freq in freq_list:
        ent += freq * np.log2(freq)
    ent = -ent

    return ent
