import numpy
import os
from dask.distributed import Client
import logging
import json
import pandas as pd
import collections

import const
import algs
import data
import consistency as consist
import hhs_utils as utils
import evaluate as eva


def store_alg_res(method, res_fn, sizehist, seed, is_round, norm, budget):
    if method == 'Hc':
        x_hat = algs.estimate_hc(sizehist, seed=seed, cum_round=is_round,
            iso_norm=norm, budget=budget)
        err = algs.evaluate_hc(sizehist, x_hat)
        print('error: ', err)

    elif method == 'Hg':
        hg_hat = algs.estimate_hg(norm, sizehist, seed=seed, budget=budget,
            is_round=is_round)
        x_hat = utils.convert_esthg_to_sizehist(hg_hat, const.public_max_size)
        err = algs.evaluate_hg(sizehist, hg_hat)
        print('Hg error: ', err)

    # store estimated sizehist
    f =  os.path.join(const.res_dir, res_fn)
    numpy.savetxt(f, x_hat, delimiter=",")

def get_regions_estimate(method, load_data_func, norm, public_max_size=100000, is_round=True,
    level='taxi_lv1', budget=1.0, seed=20):
    client = Client()
    jobs = []
    if level.split('_')[1] == 'lv1':

        sizehist = load_data_func(level)
        sizehist = sizehist[:public_max_size+1]

        histfn = 'repr-{0}_level-{1}_round-{2}_b-{3}_maxsize-{4}_seed-{5}'.format(
            method, level, is_round, budget, public_max_size, seed)

        jobs.append(client.submit(store_alg_res, method, histfn, sizehist, seed,
            is_round, norm, budget))

        if jobs:
            client.gather(jobs)

    elif level.split('_')[1] == 'lv2':
        lv2_sizehists = load_data_func(level)

        sorted_k = sorted(lv2_sizehists.keys())
        for r_idx, r_k in enumerate(sorted_k):
            logging.debug('region name: {}'.format(r_k))
            sizehist = lv2_sizehists[r_k][:public_max_size+1]

            histfn = 'repr-{0}_level-{1}_r-{2}_round-{3}_b-{4}_maxsize-{5}_seed-{6}'.format(
                method, level, r_idx, is_round, budget, public_max_size, seed)

            jobs.append(client.submit(store_alg_res, method, histfn, sizehist, seed,
                is_round, norm, budget))

            if len(jobs)==const.partition:
                client.gather(jobs)
                jobs = []

        if jobs:
            client.gather(jobs)

    elif level.split('_')[1] == 'lv3':
        lv3_sizehist_mat = load_data_func(level)
        for r in range(len(lv3_sizehist_mat)):
            for c in range(len(lv3_sizehist_mat[r])):

                r_k = '{0}-{1}'.format(r, c)
                sizehist = lv3_sizehist_mat[r][c][:public_max_size+1]

                histfn = 'repr-{0}_level-{1}_r-{2}_round-{3}_b-{4}_maxsize-{5}_seed-{6}'.format(
                    method, level, r_k, is_round, budget, public_max_size, seed)

                jobs.append(client.submit(store_alg_res, method, histfn, sizehist,
                    seed, is_round, norm, budget))


                if len(jobs)==const.partition:
                    client.gather(jobs)
                    jobs = []
        if jobs:
            client.gather(jobs)

    client.close()

def preprocess_size_var(seed, public_max_size, load_data_func, pdataset, cdataset, prepr='Hc',
    crepr='Hc', budget=1.0):

    # load the parent level
    var_func = consist.cal_Hc_var if prepr == 'Hc' else consist.cal_Hg_var

    fname = 'repr-{0}_level-{1}_round-True_b-{2}_maxsize-{3}_seed-{4}'.format(
            prepr, pdataset, budget, public_max_size, seed)
    fpath = os.path.join(const.res_dir, fname)

    p_sizehist = utils.load_arr(fpath)
    logging.debug('len of parent sizehist: {}'.format(p_sizehist.size))

    parent_sizeCnt_dic = utils.convert_sizehist_to_dic(p_sizehist, dict_type="ordered")
    total_hh = sum(parent_sizeCnt_dic.values())
    parent_sizeVar_dic = var_func(parent_sizeCnt_dic, budget)

    #only get children true size hist
    c_sizehists = load_data_func(cdataset)
    sorted_k = sorted(c_sizehists.keys())

    list_of_sizeCnt_dic = []
    list_of_sizeVar_dic = []

    for r_idx, r_k in enumerate(sorted_k):
        logging.debug('r_k: {}'.format(r_k))
        c_var_func = consist.cal_Hc_var if crepr == 'Hc' else consist.cal_Hg_var
        cfn = 'repr-{0}_level-{1}_r-{2}_round-True_b-{3}_maxsize-{4}_seed-{5}'.format(
                crepr, cdataset, r_idx, budget, public_max_size, seed)
        fpath = os.path.join(const.res_dir, cfn)

        c_sizehist = utils.load_arr(fpath)
        c_sizeCnt_dic = utils.convert_sizehist_to_dic(c_sizehist, dict_type="ordered")
        list_of_sizeCnt_dic.append(c_sizeCnt_dic)
        list_of_sizeVar_dic.append(var_func(c_sizeCnt_dic, budget))

    logging.debug('list_of_sizeVar_dic is {}'.format(list_of_sizeVar_dic))
    logging.debug('len of children: {}'.format(len(list_of_sizeCnt_dic)))

    return parent_sizeCnt_dic, list_of_sizeCnt_dic, parent_sizeVar_dic, list_of_sizeVar_dic


def init_batch_matching(seed, public_max_size, load_data_func, order='MSF',
    weight_metric='weight', pdataset='taxi_lv1', prepr='Hc',
    cdataset='taxi_lv2', crepr='Hc', budget=1.0):

    '''
    match regions at 1st level with regions at 2nd level.
    Inputs: load_data_func takes a string to specify which level in the hierarchy
        and output the histograms of all the regions in that level.
    '''
    p_sizeCnt_dic, list_of_sizeCnt_dic, p_sizeVar_dic, list_of_sizeVar_dic = preprocess_size_var(
        seed, public_max_size, load_data_func, pdataset, cdataset, prepr=prepr, crepr=crepr, budget=budget)

    if order == 'MSF':
        list_of_chHH, list_of_chHH_tup = consist.batch_opt_matching(p_sizeCnt_dic,
            list_of_sizeCnt_dic, p_sizeVar_dic, list_of_sizeVar_dic, weight_metric)
    logging.debug('======consistency done======')

    outf_prefix = 'res_from_pds-{0}_prepr-{1}_cds-{2}_crepr-{3}_w-{4}_order-{5}_s-{6}_b-{7}'.format(
        pdataset, prepr, cdataset, crepr, weight_metric, order, seed, budget)

    # add the (size, var) to the result when using weight_metric == 'weight'
    with open(os.path.join(const.res_dir, '{0}_consistSV'.format(outf_prefix)), 'w') as f:
        s = json.dumps(list_of_chHH_tup)
        f.write(s)

    # evaluate
    c_sizehists = load_data_func(cdataset)
    sorted_k = sorted(c_sizehists.keys())

    df = pd.DataFrame(columns=['region', 'emd'])
    total_root_hh = 0
    res_p_sizehist = numpy.zeros(const.public_max_size+1)
    for r_idx, r_HH in enumerate(list_of_chHH):
        res_c_sizehist = numpy.zeros(const.public_max_size+1)
        total_root_hh += len(r_HH)

        for i in range(len(r_HH)): #r_HH is a list
            res_c_sizehist[int(round(r_HH[i]))] += 1
            res_p_sizehist[int(round(r_HH[i]))] += 1

        #get true region sizehist
        r_name = sorted_k[r_idx]
        val = eva.emd(c_sizehists[r_name], res_c_sizehist)
        logging.debug('val: {}'.format(val))
        df = df.append({'region': r_idx, 'emd': val}, ignore_index=True)


    root_sizehist = load_data_func(pdataset)
    val = eva.emd(root_sizehist, res_p_sizehist)
    logging.debug('root val: {}'.format(val))

    df = df.append({'region': 'root', 'emd': val}, ignore_index=True)
    print('result: \n', df)
    df.to_csv(os.path.join(const.res_dir, '{0}_consistBatch.csv'.format(outf_prefix)), index=False)


def matching_back_substitution_across_hierarchy(seed, public_max_size, load_data_func,
    order='MSF', weight_metric='weight', pdataset='taxi', prepr='HcHc',
    cdataset='taxi_lv3', crepr='Hc', budget=1.0):

    '''
    use the consistency result from the initial matching for level 1 and level 2.
    perform the consistency matching for level 2 and level 3.
    The backward substitution generates the final consistent estimates for all regions
    in the hierarchy.

    children at level 3 is a 2d array C[i][j] where i is the parent id, j is the
    child id, C[i][j] is the numpy histogram
    '''

    list_of_chHH_tup = None
    if prepr == 'HgHg':
        fn = 'res_from_pds-{0}_lv1_prepr-Hg_cds-{1}_lv2_crepr-Hg_w-{2}_order-{3}_s-{4}_b-{5}_consistSV'.format(
            pdataset, pdataset, weight_metric, order, seed, budget)

    elif prepr == 'HcHc':
        fn = 'res_from_pds-{0}_lv1_prepr-Hc_cds-{1}_lv2_crepr-Hc_w-{2}_order-{3}_s-{4}_b-{5}_consistSV'.format(
            pdataset, pdataset, weight_metric, order, seed, budget)
    else:
        print('initial matching result is missing')
        raise

    list_of_chHH_tup = utils.load_new_est_from_file(fn)
    logging.info('======finish load======')

    df = pd.DataFrame(columns=['region', 'emd'])
    outf_prefix = 'res_from_pds-{0}_prepr-{1}_cds-{2}_crepr-{3}_w-{4}_order-{5}_s-{6}_b-{7}'.format(
        pdataset, prepr, cdataset, crepr, weight_metric, order, seed, budget)

    C = load_data_func(cdataset)
    lv2_sizehists = load_data_func('{}_lv2'.format(pdataset))
    sorted_r_k = sorted(lv2_sizehists)
    lv1_sizehist = load_data_func('{}_lv1'.format(pdataset))

    root_estsh = numpy.zeros(const.public_max_size+1)
    for r in range(len(C)):
        logging.debug('r: {}'.format(r))

        # r is the lv2 region idx
        # the wavg is float, not yet round()
        phhs = []
        parent_sizeVar_dic = {}
        parent_sv_tmp = collections.defaultdict(list)

        for (wavg, var) in list_of_chHH_tup[r]:
            phhs.append(wavg)
            rdwavg = round(wavg)
            parent_sv_tmp[rdwavg].append(var)

        # take avg
        for rdsz in parent_sv_tmp.keys():
            aggr_v = float(sum(parent_sv_tmp[rdsz]))/len(parent_sv_tmp[rdsz])
            parent_sizeVar_dic[rdsz] = aggr_v


        parent_sizeCnt_dic = consist.gen_size_cnt_dic(phhs, dict_type='default')
        total_hh = sum(parent_sizeCnt_dic.values())
        logging.debug('total # hh in parent: {}'.format(total_hh))

        # retrieve its children array
        list_of_sizeCnt_dic = []
        list_of_sizeVar_dic = []

        for c in range(len(C[r])):
            r_k = '{0}-{1}'.format(r, c)
            print('r_k: ', r_k)

            c_var_func = consist.cal_Hc_var if crepr == 'Hc' else consist.cal_Hg_var
            cfn = 'repr-{0}_level-{1}_r-{2}_round-True_b-{3}_maxsize-{4}_seed-{5}'.format(
                crepr, cdataset, r_k, budget, public_max_size, seed)
            fpath = os.path.join(const.res_dir, cfn)
            c_sizehist = utils.load_arr(fpath)

            c_sizeCnt_dic = utils.convert_sizehist_to_dic(c_sizehist, dict_type="ordered")
            list_of_sizeCnt_dic.append(c_sizeCnt_dic)
            list_of_sizeVar_dic.append(c_var_func(c_sizeCnt_dic, budget))

        if order == 'MSF':
            list_of_Lv2HH, list_of_Lv2HH_tup = consist.batch_opt_matching(parent_sizeCnt_dic,
                list_of_sizeCnt_dic, parent_sizeVar_dic, list_of_sizeVar_dic, weight_metric)
        logging.debug('======one consistency done======')

        total_ch_hh = 0
        pa_estsh = numpy.zeros(const.public_max_size+1)
        pa_true_hist = numpy.zeros(const.public_max_size+1)
        for ch_idx, ch_HH in enumerate(list_of_Lv2HH):
            ch_estsh = numpy.zeros(const.public_max_size+1)
            total_ch_hh += len(ch_HH)

            for i in range(len(ch_HH)):
                ch_estsh[int(round(ch_HH[i]))] += 1
                pa_estsh[int(round(ch_HH[i]))] += 1
                root_estsh[int(round(ch_HH[i]))] += 1


            ck = '{0}-{1}'.format(r, ch_idx)
            pa_true_hist += C[r][ch_idx]

            # write out consistency res
            consistfn = 'res_from_{0}{1}_r-{2}_w-{3}_order-{4}_s-{5}_b-{6}_consistency'.format(
                prepr, crepr, ck, weight_metric, order, seed, budget)

            numpy.savetxt(os.path.join(const.res_dir, consistfn), ch_estsh, delimiter=",")

            #to test if the res file is correct
            #consist_res = utils.load_arr('./res/{}'.format(consistfn))
            #assert numpy.array_equal(consist_res, ch_estsh)

            val = eva.emd(C[r][ch_idx], ch_estsh)
            logging.debug('ct val: {}'.format(val))
            df = df.append({'region': ck, 'emd': val}, ignore_index=True)

        assert total_ch_hh == total_hh

        # check sum of children equals parent
        assert numpy.array_equal(pa_true_hist, lv2_sizehists[sorted_r_k[r]])

        # write out consistency res
        consistfn = 'res_from_{0}{1}_r-{2}_w-{3}_order-{4}_s-{5}_b-{6}_consistency'.format(
            prepr, crepr, r, weight_metric, order, seed, budget)
        numpy.savetxt(os.path.join(const.res_dir, consistfn), pa_estsh, delimiter=",")


        val = eva.emd(lv2_sizehists[sorted_r_k[r]], pa_estsh)
        logging.debug('lv2, val: {}'.format(val))
        df = df.append({'region': r, 'emd': val}, ignore_index=True)

    consistfn = 'res_from_{0}{1}_r-root_w-{2}_order-{3}_s-{4}_b-{5}_consistency'.format(
        prepr, crepr, weight_metric, order, seed, budget)
    numpy.savetxt(os.path.join(const.res_dir, consistfn), root_estsh, delimiter=",")

    #evaluate root
    val = eva.emd(lv1_sizehist, root_estsh)
    logging.debug('root val: {}'.format(val))
    df = df.append({'region': 'root', 'emd': val}, ignore_index=True)
    print('result: \n', df)
    df.to_csv(os.path.join(const.res_dir, '{0}_consistBatch.csv'.format(outf_prefix)), index=False)
