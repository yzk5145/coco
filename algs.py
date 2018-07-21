import numpy
import os
import logging
import argparse

import combine as com
import PAV as pav
import evaluate as eva
import isotonic_reg as iso
import hhs_utils as utils
import data
import const

def estimate_hg(norm, sizehist, seed=20, budget=1.0, is_round=False):
    hg = utils.expand_sizehist_to_hg(sizehist)
    #hg is sorted
    min_size = hg[0]

    if norm == 'l1':
        prng = numpy.random.RandomState(seed)
        _, noisy_hg = com.noisy_group_sizes(sizehist, budget, prng)

        hg_hat = iso.isotonic_l1(noisy_hg, var_type='CONTINUOUS', nonnegative=True,
            starts_at=min_size, ends_at=None, weighted_total=None)
    elif norm == 'l2':
        hg_hat = pav.get_pav_estimate(sizehist, seed, budget, starts_at=min_size)

    #round the pav_Hg
    if is_round:
        hg_hat = numpy.round(hg_hat)

    return hg_hat


def estimate_hg_l1(sizehist, seed=20, budget=1.0, is_round=False):
    '''
    evaluate Hg method on histogram of a region
    L2 isotonic on Hg with gurobi solver, with min size and total groups constraints
    '''
    return estimate_hg('l1', sizehist, seed=seed, budget=budget, is_round=is_round)


def esimate_hg_l2(sizehist, seed=20, budget=1.0, is_round=False):
    '''
    evaluate Hg method on histogram of a region
    L2 isotonic on Hg with PAV, with min size and total groups constraints
    '''
    return estimate_hg('l2', sizehist, seed=seed, budget=budget, is_round=is_round)


def evaluate_hg(sizehist, hg_hat):
    '''
    evaluate hg_hat with emd, will round hg_hat
    '''
    #convert hg_hat to size histogram
    sizehist_hat = utils.convert_esthg_to_sizehist(hg_hat, const.public_max_size)
    err = eva.emd(sizehist, sizehist_hat)
    return err


def estimate_hc(sizehist, seed=20, cum_round=True, iso_norm='iso_l1', budget=1.0):
    '''
    evaluate Hc method on histogram of a region
    L1 or L2 isotonic on Hc, with min size and total groups constraints
    '''
    sizecum = sizehist.cumsum()
    ends_at = sizecum[-1]

    hg = utils.expand_sizehist_to_hg(sizehist)
    #hg is sorted
    min_size = hg[0]

    if min_size == 1:
        #dataset does not have size 0, counts at size 0 is 0
        noise_starts_at = 1
        starts_at = 0
    elif min_size == 0:
        noise_starts_at = 0
        starts_at = None
    else:
        # TODO: handle min_size > 1
        print('exception')
        raise

    prng = numpy.random.RandomState(seed)
    noisy_cum_sum = com.noisy_increasing_hist(numpy.array(sizehist), budget, prng,
        const.public_max_size+1, noise_starts_at=noise_starts_at)

    if iso_norm == 'iso_l1':
        exec_solver = iso.isotonic_l1
    elif iso_norm == 'iso_l2':
        exec_solver = iso.isotonic_l2

    cum_x = exec_solver(noisy_cum_sum, var_type='CONTINUOUS', nonnegative=True,
        starts_at=starts_at, ends_at=ends_at, weighted_total=None)

    if cum_x is None:
        sizehist_hat = None
    else:
        if cum_round:
            #need to round the cum_x, then convert it to hist
            cum_x = numpy.round(cum_x)

        sizehist_hat = utils.to_hist(cum_x)

    return sizehist_hat


def evaluate_hc(sizehist, sizehist_hat):
    err = eva.emd(sizehist, sizehist_hat)
    return err
