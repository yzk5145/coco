import mechanisms
import numpy

def noisy_hist(data, budget, prng, max_size, noise_starts_at=1):

    hist = numpy.zeros(max_size)
    if noise_starts_at == 1:
        hist[0] = 0.0
        hist[1:data.size]= mechanisms.geometric_mechanism(true_answer = data[1:], budget=budget, sensitivity=1.0, prng=prng)
    else:
        hist[0:data.size]= mechanisms.geometric_mechanism(true_answer = data, budget=budget, sensitivity=1.0, prng=prng)

    return hist

def noisy_increasing_hist(data, budget, prng, max_size, noise_starts_at=1):
    ''' Add noise to the cumulative sum of a histogram except for the last
    element which is assumed to be known (total # of groups)

    Input:
        data: a group size histogram in a numpy array
        budget: the privacy budget to use
        prng: a numpy.random.RandomState instance
        max_size: the maximum group size to use. This should be
             either known or a privacy-preserving estimate
    Output:
        hist: the noisy cumulative sum histogram

    '''
    myhist = numpy.zeros(max_size)
    myhist[0:data.size]=data
    cumsum = myhist.cumsum()
    hist = numpy.zeros(cumsum.size)

    if noise_starts_at == 1:
        #dataset does not have size 0
        hist[0]=0.0
        hist[-1] = cumsum[-1]
        hist[1:hist.size-1] = mechanisms.geometric_mechanism(true_answer =
            cumsum[1:hist.size-1], budget=budget, sensitivity=1.0, prng=prng)

    elif noise_starts_at == 0:
        hist[-1] = cumsum[-1]
        hist[0:hist.size-1] = mechanisms.geometric_mechanism(true_answer =
            cumsum[0:hist.size-1], budget=budget, sensitivity=1.0, prng=prng)
    else:
        #todo- if there are datasets whose min size > 1
        print('unknown noise_starts_at')
        raise

    return hist

def noisy_group_sizes(data, budget, prng):
    '''
    Adds noise to each group

    Inputs:
        data: a numpy array where data[i] is the number of groups of size i
        budget: privacy budget
        prng: a numpy RandomState instance
    Outputs:
        hist: a dictionary where h[i] is the number of groups of size i
        arr_hist: a numpy array of the noisy group sizes. It is guaranteed that
             the original group sizes corresponding to them are in nondecreasing
             order.
    Notes:
        the noisy group sizes can be negative, but they will be integers
    '''

    hist = {}
    index = 0
    out_index = 0
    arr_hist = numpy.zeros(data.sum())
    for (index, count) in enumerate(data):

        for i in range(count):
            noisy_group_size = mechanisms.geometric_mechanism(true_answer=index,
                budget=budget, sensitivity=1.0, prng=prng)
            arr_hist[out_index] = noisy_group_size
            out_index = out_index + 1
            hist[noisy_group_size] = hist.get(noisy_group_size, 0) + 1
    return hist, arr_hist
