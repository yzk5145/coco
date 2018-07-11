import numpy
import os
import const

data_dir = './data/'


def retrieve_taxi_sizehist(dataset):
    #taxi_level*.npy first counts corresponds to index 1
    if dataset == 'taxi_lv1':
        fn = 'taxi_level1.npy'
        sizehist = numpy.load(os.path.join(data_dir, fn))
        my_sizehist = numpy.zeros(const.public_max_size + 1, dtype=numpy.int32)
        my_sizehist[1: sizehist.size+1] = sizehist

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
