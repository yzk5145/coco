import numpy


def compatibility_resize(a, b):
    len_a = numpy.size(a)
    len_b = numpy.size(b)

    newa = a.copy()
    newb = b.copy()
    if len_a > len_b:
        newb.resize(len_a)

    elif len_b > len_a:
        newa.resize(len_b)
    return newa, newb

def emd(correct, estimated):
    newa, newb = compatibility_resize(correct, estimated)
    diff = newa.sum() - newb.sum()

    if diff > 0:
        newb[0] = abs(diff)
        print("true total household > estimated")
    elif diff < 0:
        newa[0] = abs(diff)
        print("true total household FEWER estimated")

    ca = newa.cumsum()
    cb = newb.cumsum()

    return float(numpy.abs(ca - cb).sum())
