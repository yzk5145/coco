import numpy


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
