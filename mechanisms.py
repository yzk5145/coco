import numpy


def laplace_mechanism(*, true_answer, budget, sensitivity, prng):
    shape = numpy.shape(true_answer)
    return true_answer + prng.laplace(scale=sensitivity/float(budget), size=shape)

def geometric_mechanism(*, true_answer, budget, sensitivity, prng):

    shape = numpy.shape(true_answer)
    epsilon = budget / float(sensitivity)
    p = 1 - numpy.exp(-epsilon) #alpha = numpy.exp(-epsilon)
    x = prng.geometric(p, size=shape) - 1 #numpy geometrics start with 1
    y = prng.geometric(p, size = shape) - 1
    return x-y + true_answer
