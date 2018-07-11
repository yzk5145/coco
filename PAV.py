import os
import numpy

from sklearn.isotonic import IsotonicRegression
import synthdata as syn
import combine as com
import evaluate as eva
import hhs_utils as utils
import const


def get_pav_estimate(sizehist, seed, budget, starts_at=None):
    """
    L2 isotonic regression
    every entry >= 1
    no max size constraint
    """

    if starts_at is not None:
        starts_at = starts_at


    prng = numpy.random.RandomState(seed)
    _, noisy_Hg = com.noisy_group_sizes(sizehist, budget, prng)

    ir = IsotonicRegression(y_min=starts_at)
    pav_x = ir.fit_transform(range(len(noisy_Hg)), noisy_Hg)

    return pav_x
