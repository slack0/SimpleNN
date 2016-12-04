from utilities import linear
import numpy as np


class Layer(object):
    '''
    base class for layer objects
    '''

    def __init__(self, incoming):
        '''
        INPUT: 
            incoming: tuple or Layer object
        OUTPUT: None

        initialize Layer object
        '''
        if isinstance(incoming, tuple):
            self.input_shape = incoming
            self.input_layer = None
        else:
            self.input_shape = incoming.output_shape()
            self.input_layer = incoming

    def output_shape(self):
        '''
        INPUT: None
        OUTPUT: tuple

        return shape of layer output
        '''
        return self._get_output_shape_for(self.input_shape)

    def output(self, input):
        '''
        INPUT: 
            input: numpy array
        OUTPUT: numpy array

        take incoming data or result of previous layer and return layer output
        '''
        return self._get_output_for(input)

    def _get_output_shape_for(self, input_shape):
        '''
        INPUT: 
            input_shape: tuple 
        OUTPUT: tuple 
        function for computing output shape from input shape, should be 
        overwritten for different kinds of layers
        '''
        return input_shape

    def _get_weighted_input(self, input):
        '''
        INPUT:
            input: numpy array
        OUTPUT: numpy array

        function for computing weighted input of the layer, overwrite for different
        kinds of layers
        '''
        return input

    def _get_output_for(self, input):
        '''
        INPUT:
            input: numpy array
        OUTPUT: numpy array

        function for computing output of layer based on incoming data
        '''
        return input


class DenseLayer(Layer):
    '''
    Layer which connects all nodes of the previous layer to each node in the
    current layer
    '''

    def __init__(self, incoming, num_units, nonlinearity, bias=True, **kwargs):
        '''
        INPUT:
            incoming: Layer object
            num_units: float
            nonlinearity: function
            bias: bool
        OUTPUT: None

        initialize layer object, takes in previous layer, number of units in layer,
        and the nonlinear function to apply
        '''
        super(DenseLayer, self).__init__(incoming, **kwargs)
        self.nl = nonlinearity

        self.num_units = num_units

        num_input = int(np.prod(self.input_shape[1:]))

        self.W = self._make_weights(num_input)
        if bias:
            self.b = np.zeros((1, num_units))
        else:
            self.b = None

    def _make_weights(self, size_in):
        '''
        INPUT:
            size_in: float, number of nodes in previous layer
        OUTPUT: numpy array of weights

        initialize weights for layer using Glorot Initialization
        '''
        w_shape = (size_in, self.num_units)
        vals = np.random.normal(
                    loc=0,
                    scale=np.sqrt(6. / sum(w_shape)),
                    size=w_shape
                )
        return vals

    def _get_output_shape_for(self, input_shape):
        '''
        INPUT: 
            input_shape: tuple 
        OUTPUT: tuple 

        function for computing output shape from input shape
        '''

        return (input_shape[0], self.num_units)

    def _get_output_for(self, input):
        '''
        INPUT:
            input: numpy array
        OUTPUT: numpy array

        compute output of layer for provided input
        '''
        return self.nl(self._get_weighted_input(input))

    def _get_weighted_input(self, input):
        '''
        INPUT:
            input: numpy array of data
        OUTPUT:numpy array of activations (i.e., pre-activations before applying nonlinearity)

        compute activation for layer by multiplying input by the weights
        '''

        weighted_input = np.dot(input, self.W)
        if self.b is not None:
            weighted_input += self.b

        return weighted_input

    def _update_weights(self, value):
        '''
        INPUT: 
            value: numpy array of gradients
        OUTPUT: None

        update the weights by subtracting the gradient
        '''
        self.W = self.W - value

    def _update_bias(self, value):
        '''
        INPUT: 
            value: numpy array of gradients
        OUTPUT: None

        update the bias by subtracting the gradient
        '''
        self.b = self.b - value.reshape(self.b.shape[0], self.b.shape[1])


class InputLayer(Layer):
    '''
    class for input layers, just passes the input directly through as output
    '''

    def __init__(self, shape):
        '''
        INPUT:
            shape: tuple
        OUTPUT: None

        takes in tuple of data shape, returns itself
        '''
        super(InputLayer, self).__init__(shape)
        self.shape = shape
        self.nl = linear
