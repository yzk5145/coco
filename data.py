import numpy
import os
import const

data_dir = './data/'


def retrieve_taxi_sizehist(dataset):
    #taxi_level*.npy first counts corresponds to index 1
    if dataset == 'taxi_lv1': #manhattan
        fn = 'taxi_level1.npy'
        sizehist = numpy.load(os.path.join(data_dir, fn))
        my_sizehist = numpy.zeros(const.public_max_size + 1, dtype=numpy.int32)
        my_sizehist[1: sizehist.size+1] = sizehist

    elif dataset == 'taxi_lv2': #town
        fn = 'taxi_level2.npy'
        #row 0 is lower manh

        tmp = numpy.load(os.path.join(data_dir, fn))
        towns = {}

        for r in range(tmp.shape[0]):
            #print('tmp[r]: ', tmp[r]) #[578 412 362 ...   0   0   1]
            tn_sizehist = numpy.zeros(const.public_max_size + 1, dtype=numpy.int32)
            tn_sizehist[1: tmp[r].size+1] = tmp[r]
            if r == 0:
                towns['lower'] = tn_sizehist
            if r == 1:
                towns['upper'] = tn_sizehist
        return towns

    elif dataset == 'taxi_lv3': #nei
        #0~17 is lower manhattan
        fn = 'taxi_level3.npy'
        my_neihoods = [ [] for i in range(2)]

        tmp = numpy.load(os.path.join(data_dir, fn))

        for r_idx in range(tmp.shape[0]):
            nei_sizehist = numpy.zeros(const.public_max_size + 1, dtype=numpy.int32)
            nei_sizehist[1: tmp[r_idx].size+1] = tmp[r_idx]
            if r_idx <= 17:
                my_neihoods[0].append(nei_sizehist)
            else:
                my_neihoods[1].append(nei_sizehist)

        return my_neihoods
    else:
        print('unknown dataset error')
        raise

    return my_sizehist


def retrieve_race_sizehist(dataset):
    f_dic = {'white_lv1': 'White_nation_H.npy', 'asian_lv1': 'Asian_nation_H.npy',
            'black_lv1': 'Black_nation_H.npy', 'haw_lv1': 'Hawaiian_nation_H.npy',
            'native_lv1': 'Native_nation_H.npy'}

    #*_nation_H.npy: first counts corresponds to index 0
    fn = f_dic[dataset]
    sizehist = numpy.load(os.path.join(data_dir, fn))
    my_sizehist = numpy.zeros(const.public_max_size + 1, dtype=numpy.int32)
    my_sizehist[0: sizehist.size] = sizehist

    return my_sizehist
