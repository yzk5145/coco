import collections
import math
import copy
import numpy
from operator import attrgetter
import logging

def cal_Hg_var(sizeCnt_dic, budget):
    """
       var(size) = var(noise)/|Si|
       sizeCnt_dic - key: size, val: counts where counts > 0
    """
    sizeVar_dic = {}
    for k, v in sizeCnt_dic.items():
        sizeVar_dic[k] = (2.0/(budget*budget))/v
    return sizeVar_dic

def cal_Hc_var(sizeCnt_dic, budget):
    """
       var(size) = 2*var(noise)/|Si|*size
       sizeCnt_dic - key: size, val: counts where counts > 0
    """
    sizeVar_dic = {}
    for k, v in sizeCnt_dic.items():
    #for race data k contains 0
        if k == 0:
            sizeVar_dic[k] = (4.0/(budget*budget))/v
        else:
            #sizeVar_dic[k] = (4.0/(budget*budget))/v*k
            sizeVar_dic[k] = (4.0/(budget*budget))/v
    return sizeVar_dic

def aggr_count(list_of_sizeCnt_dic_cp):
    total_sizeCnt_dic = collections.defaultdict(int)

    for dic in list_of_sizeCnt_dic_cp:
        for s, c in dic.items():
            total_sizeCnt_dic[s] = total_sizeCnt_dic.get(s, 0)+c
    return total_sizeCnt_dic

def gen_size_cnt_dic(sorted_hg, dict_type='ordered'):
    size_cnt_dic = collections.OrderedDict() if dict_type=="ordered" else collections.defaultdict(int)

    for v in sorted_hg:
        size_cnt_dic[round(v)] = size_cnt_dic.get(round(v), 0)+1

    return size_cnt_dic

def new_estimate_batch_hh(matched_num, psize, p_Si, st_idx, st_size, st_Ti,
    list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, weight_metric, p_Si_cnt=None, st_Ti_cnt=None, list_of_chHH_tup=None):

    if matched_num == 0:
        return list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, list_of_chHH_tup

    if weight_metric == None:
        #assign the parent number to children
        list_of_chHH[st_idx] += [psize]*matched_num

    elif weight_metric == 'child':
        list_of_chHH[st_idx] += [st_size]*matched_num #assign children size

    elif weight_metric == 'avg':
        list_of_chHH[st_idx] += [float(psize+st_size)/2]*matched_num

    elif weight_metric == 'weight':

        #new est = (a/v1 + b/v2) / (1/v1 + 1/v2)
        logging.debug('psize: {}'.format(psize))
        logging.debug('st_size: {}'.format(st_size))
        logging.debug('p_Si: {}'.format(p_Si))
        logging.debug('st_Ti: {}'.format(st_Ti))

        size = (float(psize)/p_Si + float(st_size)/st_Ti)/(1.0/p_Si + 1.0/st_Ti)

        if p_Si_cnt is not None and st_Ti_cnt is not None:
            w = float(p_Si_cnt)/(p_Si_cnt+st_Ti_cnt)
            old_wavg = w*psize + (1-w)*st_size

            assert round(old_wavg) == round(size)
            list_of_chHH[st_idx] += [old_wavg]*matched_num
        else:
            list_of_chHH[st_idx] += [size]*matched_num

        #new var = 1/(1/v1 + 1/v2)
        var = 1.0/(1.0/p_Si + 1.0/st_Ti)
#        list_of_chHH_tup[st_idx] += [(psize, st_size, size, var)]*matched_num
        list_of_chHH_tup[st_idx] += [(size, var)]*matched_num

    # remove that matched household from state
    list_of_sizeCnt_dic[st_idx][st_size] -= matched_num
    if list_of_sizeCnt_dic[st_idx][st_size] == 0:
        del list_of_sizeCnt_dic[st_idx][st_size]

    total_sizeCnt_dic[st_size] -= matched_num
    if total_sizeCnt_dic[st_size] == 0:
        del total_sizeCnt_dic[st_size]

    return list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, list_of_chHH_tup


def batch_opt_matching(parent_sizeCnt_dic, list_of_sizeCnt_dic, parent_sizeVar_dic, list_of_sizeVar_dic, weight_metric, test_mode=False):

    if test_mode:
        list_of_sizeCnt_dic_cp = copy.deepcopy(list_of_sizeCnt_dic)
        parent_sizeCnt_dic_cp = copy.deepcopy(parent_sizeCnt_dic)

    total_sizeCnt_dic = aggr_count(list_of_sizeCnt_dic)
    list_of_chHH = [[] for _ in range(len(list_of_sizeCnt_dic))]
    list_of_chHH_tup = [[] for _ in range(len(list_of_sizeCnt_dic))]

    logging.debug('total number of sizes in parent {}'.format(len(parent_sizeCnt_dic)))

    cost_sum = 0
    while len(parent_sizeCnt_dic) > 0 :
        key = min(parent_sizeCnt_dic.keys())
        for psize in [key]:
            pcnt = parent_sizeCnt_dic[psize]
            tsz = min(total_sizeCnt_dic.keys())
            total_chh = 0

            if tsz in total_sizeCnt_dic.keys():
                total_chh += total_sizeCnt_dic[tsz]

            target_sizes = [tsz]
            #2 cases
            if pcnt >= total_chh:
                # match all hh of the children
                for st_idx, dic in enumerate(list_of_sizeCnt_dic):
                    for tsz in target_sizes:
                        if tsz in dic.keys():
                            matched_num = list_of_sizeCnt_dic[st_idx][tsz]
                            st_Ti = list_of_sizeVar_dic[st_idx][tsz]
                            st_size = tsz
                            p_Si = parent_sizeVar_dic[psize]

                            if test_mode:
                                st_Ti_cnt = list_of_sizeCnt_dic_cp[st_idx][tsz]
                                p_Si_cnt = parent_sizeCnt_dic_cp[psize]
                                list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, list_of_chHH_tup = new_estimate_batch_hh(matched_num, psize, p_Si,
                                    st_idx, st_size, st_Ti, list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, weight_metric, p_Si_cnt, st_Ti_cnt, list_of_chHH_tup)
                            else:
                                cost_sum += (abs(psize-st_size)*matched_num)
                                list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, list_of_chHH_tup = new_estimate_batch_hh(matched_num, psize, p_Si,
                                    st_idx, st_size, st_Ti, list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, weight_metric, None, None, list_of_chHH_tup)

                            parent_sizeCnt_dic[psize] -= matched_num
                            if parent_sizeCnt_dic[psize] == 0:
                                del parent_sizeCnt_dic[psize]

            else:

                Float = collections.namedtuple("Float", ["fraction", "ratio", "stID", "sz"])
                float_entries = []
                lb = 0
                for st_idx, dic in enumerate(list_of_sizeCnt_dic):
                    for tsz in target_sizes:
                        if tsz in dic.keys():
                            hh = dic[tsz]
                            if hh > 0 :
                                ratio = pcnt*(float(hh)/total_chh)
                                if ratio.is_integer():
                                    lb += ratio
                                else:
                                    lb += math.floor(ratio)
                                float_entries.append(Float(math.modf(ratio)[0], ratio, st_idx, tsz)) ##[0] is fractional part, [1] is integer part

                float_entries.sort(key=attrgetter('fraction'), reverse=True) #decreasing order
                optimal_k = int(pcnt - lb)

                if optimal_k > len(float_entries):
                    #print('======round every floats up still not reach pcnt======')
                    raise

                for i in range(len(float_entries)):
                    #round top k up
                    if i in range(optimal_k):
                        assign_num = math.ceil(float_entries[i].ratio)
                    else:
                        assign_num = math.floor(float_entries[i].ratio)

                    if assign_num == 0:
                        continue
                    else:
                        st_Ti = list_of_sizeVar_dic[float_entries[i].stID][float_entries[i].sz]
                        st_size = float_entries[i].sz
                        p_Si = parent_sizeVar_dic[psize]

                        if test_mode:
                            st_Ti_cnt = list_of_sizeCnt_dic_cp[float_entries[i].stID][float_entries[i].sz]
                            p_Si_cnt = parent_sizeCnt_dic_cp[psize]

                            list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, list_of_chHH_tup = new_estimate_batch_hh(assign_num, psize, p_Si,
                                    float_entries[i].stID, st_size, st_Ti, list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, weight_metric, p_Si_cnt, st_Ti_cnt, list_of_chHH_tup)
                        else:
                            cost_sum += (abs(psize-st_size)*assign_num)
                            list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, list_of_chHH_tup = new_estimate_batch_hh(assign_num, psize, p_Si,
                                float_entries[i].stID, st_size, st_Ti, list_of_sizeCnt_dic, list_of_chHH, total_sizeCnt_dic, weight_metric, None, None, list_of_chHH_tup)

                        parent_sizeCnt_dic[psize] -= assign_num
                        if parent_sizeCnt_dic[psize] == 0:
                            del parent_sizeCnt_dic[psize]

    logging.debug('cost_sum: {}'.format(cost_sum))
    return list_of_chHH, list_of_chHH_tup
