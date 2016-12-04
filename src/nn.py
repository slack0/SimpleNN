import numpy as np

from time import time

import re

from utilities import negative_log_likelihood, mean_squared, softmax, make_output


class NeuralNet(object):
    '''
    class for ccreating neural networks, takes a list of layers as input and
    creates a neural net that can be passed data and fit
    '''

    def __init__(self,
                 layers,
                 batch_size=128,
                 val_size=0.2,
                 learning_rate=0.01,
                 regression=True,
                 cost=None
                 ):
        '''
        INPUT:
            layers: list of layer objects
            batch_size: int
            val_size: float
            learning_rate: float
            regression: bool
            cost: function or None
        OUTPUT: None
        '''
        self.eta = learning_rate
        self.mod = learning_rate / float(batch_size)
        self.val_size = val_size

        self.regression = regression

        if regression and layers[-1].nl != linear:
            raise AssertionError

        if regression:
            if cost is None:
                self.natural_cost = True
                self.cost = mean_squared
        elif layers[-1].nl.__name__ == 'softmax' and cost is None:
            self.natural_cost = True
            self.cost = negative_log_likelihood
        elif layers[-1].nl.__name__ == 'sigmoid' and cost is None:
            self.natural_cost = True
            self.cost = binary_cross_entropy

        if not self.natural_cost:
            raise AssertionError

        self.layers = layers
        self.num_layers = len(layers)

        self.batch_size = batch_size
        self.num_params = 0
        self._propogate_batch_size()

    def describe(self):
        '''
        INPUT: None
        OUTPUT: None

        print the sizes and activation functions of the neural network
        '''
        for i, layer in enumerate(self.layers):
            shape_1 = layer.input_shape
            if i == 0:
                print 'layer 0: {} with linear activation'.format(shape_1)
            else:
                shape_2 = layer.W.shape
                s = [shape_1, shape_2, (shape_1[0], shape_2[1])]
                out = 'layer {}: {} * {} = {} with {} activation'
                print out.format(i, s[0], s[1], s[2], layer.nl.__name__)

    def _propogate_batch_size(self):
        '''
        INPUT: None
        OUTPUT: None

        set batch size for all layers in the network
        '''
        for i, layer in enumerate(self.layers):
            layer.input_shape = (self.batch_size, layer.input_shape[1])
            if i > 0:
                shape = layer.W.shape
                self.num_params += shape[0] * shape[1] + shape[1]

    def _forward_pass(self, input):
        '''
        INPUT:
            input: numpy array of input data each row is an example
        OUTPUT: tuple (activations, weighted inputs)

        compute weighted_inputs and activations for the network, save them both
        layer by layer
        '''
        activations = [input]
        weighted_inputs = []
        for i, layer in enumerate(self.layers):
            # print('Computing weighted input, activations for layer: {}'.format(i))
            weighted_input = layer._get_weighted_input(activations[-1])
            weighted_inputs.append(weighted_input)
            activations.append(layer.nl(weighted_input))

        return (activations, weighted_inputs)

    def eval(self, x):
        '''
        INPUT:
            x: numpy array of input data
        OUTPUT: numpy array

        get result of a single forward pass through the network
        '''
        return self._forward_pass(x)[0][-1]

    def score(self, x=None, y=None, mode=None):
        '''
        INPUT:
            x: numpy array of input data or None
            y: numpy array of labels or None
        OUTPUT: float

        return mean squared error or accuracy of network
        '''
        if x is None:
            x = self.eval(self.X)
            y = self.y

        if self.regression or mode == 'cost':
            out = self.cost(x, y)
        else:
            x = np.argmax(x, axis=1)
            y = np.argmax(y, axis=1)
            out = np.sum(x == y) / float(y.shape[0])
        return out

    def _print_epoch(self, index, train_cost, start):
        '''
        INPUT:
            index: int
            train_cost: float
            start: timestamp
        OUTPUT: None

        prints the epoch by epoch updates of costs,
        index is the current training epoch, 
        train cost is the current train cost for the network, 
        and start is the time training started for the current epoch
        '''
        result = self.eval(self.val_X)
        val_cost = self.cost(result, self.val_y)
        ratio = train_cost / val_cost
        outputs = [index, train_cost, val_cost, ratio]
        if not self.regression:
            outputs.append(self.score(result, self.val_y))
        outputs.append(time() - start)
        print self.out.format(*outputs)

    def _backprop(self, input, target):
        '''
        INPUT:
            input: numpy array of input data each row is an example
            target: numpy array of target labels or values (each row is the 
                label of the coresponding intput)
        OUTPUT: float (training cost)

        takes in data and targets and runs backpropagation to update the weights
        in the network. returns the current training cost
        '''
        ys, xs = self._forward_pass(input)

        weight_updates = [np.zeros_like(l.W) for l in self.layers[1:]]
        bias_updates = [np.zeros_like(l.b) for l in self.layers[1:]]

        delta = (ys[-1] - target)
        layer = self.layers[-1]

        weight_updates[-1] = self.eta * np.dot(ys[-2].T, delta)
        bias_updates[-1] = self.mod * np.sum(delta, axis=0)

        for l in xrange(2, self.num_layers):
            prev_layer = layer
            layer = self.layers[-l]
            delta = np.dot(delta, prev_layer.W.T) * layer.nl(xs[-l], True)
            weight_updates[-l] = self.eta * np.dot(ys[-l - 1].T, delta)
            bias_updates[-l] = self.mod * np.sum(delta, axis=0)

        for l in range(1, self.num_layers):
            self.layers[-l]._update_bias(bias_updates[-l])
            self.layers[-l]._update_weights(weight_updates[-l])

        return self.cost(ys[-1], target)

    def fit(self, train, val=None, max_iters=3):
        '''
        INPUT:
            X: numpy array (n by k) n is number of rows, k is number of features
            y: numpy array (n by 1) of labels
            val_x: numpy array or None
            val_y: numpy array or None
            max_iters: int, maximum number of epochs to train
        OUTPUT: None


        Fit neural network using training data. if val_x and val_y are None, then
        validation data and labels are created from the input data and labels.
        '''

        self._load_data(train, val)

        minibatch_indices = self._make_minibatches()

        self._make_labels()

        epoch = 0
        cur_train = self.score(mode='cost')
        self._print_epoch(epoch, cur_train, time())

        while (epoch < max_iters):
            self._shuffle_data()
            epoch += 1
            st = time()
            cur_train = self._fit_one(minibatch_indices)
            self._print_epoch(epoch, cur_train, st)

    def _make_minibatches(self):
        '''
        INPUT: None
        OUTPUT: x and y indices for minibatches

        create minibatch labels for training
        '''
        start_inds = np.array(range(0, self.len_X, self.batch_size))
        end_inds = start_inds + self.batch_size
        return zip(start_inds, end_inds)

    def _make_labels(self):
        '''
        INPUT: None
        OUTPUT: None

        makes the output string for printing the per epoch updates
        '''
        params = '... training network with {} learnable parameters'
        print params.format(self.num_params)

        labels = ['epoch:', 'train cost:', 'valid cost:', 'train/valid:']

        if not self.regression:
            labels.append('valid acc:')
        labels.append('time:')

        self.out = make_output(labels)

    def _load_data(self, train, val):
        '''
        INPUT:
            train: training data
            val: validation data
        OUTPUT: None
        '''

        self.X = train[0]
        self.y = train[1]
        self.len_X = self.X.shape[0]
        self._shuffle_data()

        if val is None:
            start = int(self.len_X * (1 - self.val_size))
            val_X = self.X[start:]
            self.X = self.X[:start]
            val_y = self.y[start:]
            self.y = self.y[:start]
        else:
            val_X = val[0]
            val_y = val[1]

        self.val_X = val_X
        self.val_y = val_y
        self.len_X = self.X.shape[0]

    def _fit_one(self, indices):
        '''
        run backpropagation on one epoch of training data
        '''
        costs = []
        for start, stop in indices:
            x = self.X[start:stop]
            y = self.y[start:stop]
            costs.append(self._backprop(x, y))

        return np.mean(costs)

    def _shuffle_data(self):
        inds = np.random.permutation(self.len_X)
        self.X = self.X[inds]
        self.y = self.y[inds]

if __name__ == '__main__':
    pass
