import numpy
import os
import logging
import argparse

import data
from algs import *

if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s,%(msecs)d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
        datefmt='%d-%m-%Y:%H:%M:%S', level=logging.INFO)

    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('-m', '--mode',
        type = str,
        choices = ['isoHc-l1', 'isoHc-l2', 'isoHg-l2', 'isoHg-l1'],
        default = 'isoHc-l1',
        help = 'default is \'isoHc\': use Hc method \n'
                '\'isoHc-l1\': Hc method with l1 postprocessing\n'
                '\'isoHc-l2\': Hc method with l2 postprocessing\n'
                '\'isoHg-l2\': Hg method with l2 postprocessing \n'
                '\'isoHg-l1\': Hg method with l1 postprocessing \n'
                )
    parser.add_argument('-b', '--budget',
        type = float,
        default = 1.0,
        help = 'default is 1.0')
    parser.add_argument('-d', '--dataset',
        type = str,
        choices = ['taxi_lv1', 'haw_lv1'],
        default = 'taxi_lv1',
        help = 'default is taxi dataset at top level')

    args = parser.parse_args()
    budget, mode, dataset = args.budget, args.mode, args.dataset

    #load data
    if dataset == 'taxi_lv1':
        sizehist = data.retrieve_taxi_sizehist(dataset)
    elif dataset == 'haw_lv1':
        sizehist = data.retrieve_race_sizehist(dataset)

    if mode == 'isoHg-l2':
        hg_hat = esimate_hg_l2(sizehist, seed=20, is_round=True, budget=budget)
        err = evaluate_hg(sizehist, hg_hat)



    elif mode == 'isoHc-l1':
        sizehist_hat = estimate_hc(sizehist, seed=20, cum_round=True,
            iso_norm='iso_l1', budget=budget)
        err = evaluate_hc(sizehist, sizehist_hat)

    elif mode == 'isoHc-l2':
        sizehist_hat = estimate_hc(sizehist, seed=20, cum_round=True,
            iso_norm='iso_l2', budget=budget)
        err = evaluate_hc(sizehist, sizehist_hat)

    elif mode == 'isoHg-l1':
        #will take very long time for hawaiian data
        hg_hat = estimate_hg_l1(sizehist, seed=20, is_round=True, budget=budget)
        err = evaluate_hg(sizehist, hg_hat)

    print('error: ', error)
