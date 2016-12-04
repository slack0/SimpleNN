import numpy as np

import os
import gzip
import cPickle
import re


def sigmoid(x, deriv=False):
    '''
    INPUT:
        x: numpy array
        deriv: bool
    OUTPUT: numpy array

    calculate the sigmoid function for the input array, if deriv is True
    calculate the derivative of sigmoid instead
    '''
    if deriv:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1. / (1. + np.exp(-x))


def tanh(x, deriv=False):
    '''
    INPUT:
        x: numpy array
        deriv: bool
    OUTPUT: numpy array

    calculate tanh nonlinearity for the input
    if deriv is True, return the derivative of tanh
    '''
    out = np.tanh(x)
    if deriv:
        return (1 - np.square(out))
    return out


def relu(x, deriv=False):
    '''
    INPUT:
        x: numpy array
        deriv: bool
    OUTPUT: numpy array

    Return the ReLU of the input numpy array
    derivative of ReLU could run into some numerical stability issues
    '''
    EPS = 1E-18
    '''
    ReLU discontinuous at 0.
    Check for miniscule values for numerical stability.
    Gradient is 0 at 0
    '''
    if deriv:
        return (x >= EPS).astype(np.int)
    return np.maximum(x, 0, x)


def softmax(x, deriv=False):
    '''
    INPUT:
        x: numpy array
        deriv: bool
    OUTPUT: numpy array

    calculate the softmax function for the input array, if deriv is True
    return an error as derivative is not implemented for this function
    The max of the input data is subtracted to prevent overflow errors
    '''
    if deriv:
        return NotImplementedError
    x = x - x.max(axis=1).reshape(-1, 1)
    e = np.exp(x)
    total = np.sum(e, axis=1)
    out = np.divide(e, total.reshape(-1, 1))
    return out


def linear(x, deriv=False):
    '''
    INPUT:
        x: numpy array
        deriv: bool
    OUTPUT: numpy array

    linear activation function, used for input data and for output of regression
    networks. if deriv is True return array of ones
    '''
    if deriv:
        return np.ones_like(x)
    return x


# cost functions
def mean_squared(y, t):
    '''
    INPUT:
        y: numpy array, (n by 1) predicted labels
        t: numpy array, (n by 1) true labels
    OUTPUT: float, mean squared error
    '''
    val = 1 / 2. * np.sum((t - y) ** 2)
    return val / y.shape[0]


def negative_log_likelihood(y, t):
    '''
    INPUT:
        y: numpy array, (n by 1) predicted labels
        t: numpy array, (n by 1) true labels
    OUTPUT: float, negative log likelihood
    '''
    val = 0
    eps = 1E-18
    y = y.clip(min=eps, max=1 - eps)
    val = -1. * np.sum(t * np.log(y))
    return val / float(y.shape[0])


def binary_cross_entropy(y, t):
    '''
    INPUT:
        y: numpy array, (n by 1) predicted labels
        t: numpy array, (n by 1) true labels
    OUTPUT: float, binary cross entropy
    '''
    val = -1. * np.sum(t * np.log(y) + (1 - t) * np.log((1 - y)))
    return val / y.shape[0]


def make_output(labels):
    '''
    INPUT: list of labels
    OUTPUT: string

    creates format string for printing per epoch updates, also prints header of
    same
    '''
    out = '  '.join(['[(): >{}]' for _ in labels])
    out = out.format(*[len(x) if i == 0 else len(x) + 2 for i, x in enumerate(labels)]).replace('()', '{}')
    out = out.format(*[i for i in range(len(labels))])
    out = out.replace('[', '{').replace(']', '}')
    print out.format(*labels)
    out = re.sub('([1,2,3,4,5]): >(\d*)', '\\1: >\\2.5f', out)
    oline = str(len(labels[-1]) + 2) + '.5'
    nline = str(len(labels[-1]) + 2) + '.3'
    out = out.replace(oline, nline)
    return out


def load_data(dataset, OneHotifyY=False):
    data_dir, data_file = os.path.split(dataset)
    if data_dir == "" and not os.path.isfile(dataset):
        new_path = os.path.join(os.path.split(__file__)[0], "../data", dataset)
        if os.path.isfile(new_path) or data_file == "mnist.pkl.gz":
            dataset = new_path
    if (not os.path.isfile(dataset)) and data_file == "mnist.pkl.gz":
        import urllib
        origin = ('http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz')
        print 'Downloading data from {}'.format(origin)
        d_path = '/'.join(os.path.split(dataset)[:-1])
        if not os.path.isdir(d_path):
            os.mkdir(d_path)
        urllib.urlretrieve(origin, dataset)

    print '... loading data'

    with gzip.open(dataset, 'rb') as f:
    # f = gzip.open(dataset, 'rb')
        train_set, valid_set, test_set = cPickle.load(f)
    # f.close()

    def OneHotify(y):
        out = np.zeros((y.shape[0], 10))
        for i, entry in enumerate(y):
            out[i][entry] = 1.
        return out

    train_x = train_set[0]
    valid_x = valid_set[0]
    test_x = test_set[0]

    if OneHotifyY:
        train_y = OneHotify(train_set[1])
        valid_y = OneHotify(valid_set[1])
        test_y = OneHotify(test_set[1])
    else:
        train_y = train_set[1]
        valid_y = valid_set[1]
        test_y = test_set[1]

    rval = ((train_x, train_y),
            (valid_x, valid_y),
            (test_x, test_y))

    return rval
