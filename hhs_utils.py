import numpy
import pandas as pd
import collections
import os
import json

import const

def expand_sizehist_to_hg(sizehist):
    hg = []
    for (index, count) in enumerate(sizehist):
        if count > 0:
            hg += [index]*int(count)
    return hg

def to_hist(infer_cum_x):
    infer_hist_x = infer_cum_x[0]
    infer_hist_x = numpy.append(infer_hist_x, numpy.diff(infer_cum_x))
    return infer_hist_x

def convert_esthg_to_sizehist(hg_hat, sizehist_maxsize):
    sizehist_hat = numpy.zeros(sizehist_maxsize+1)
    for i in range(hg_hat.size):
        if hg_hat[i].is_integer():
            sizehist_hat[int(hg_hat[i])] += 1
        else:
            sizehist_hat[int(round(hg_hat[i]))] += 1
    return sizehist_hat

def load_arr(fpath):
    df = pd.read_csv(fpath, header=None)
    arr = df[df.columns[0]].values
    return arr

def convert_sizehist_to_dic(arr, dict_type="ordered"):
    size_cnt_dic = collections.OrderedDict() if dict_type=="ordered" else collections.defaultdict(int)

    for idx, cnt in enumerate(arr):
        if cnt > 0:
            size_cnt_dic[idx] = int(size_cnt_dic.get(idx, 0)+cnt)
    return size_cnt_dic

def load_new_est_from_file(fn):

    with open(os.path.join(const.res_dir, fn)) as f:
        tmp = f.read()
        list_of_chHH_tup = json.loads(tmp)
    return list_of_chHH_tup
