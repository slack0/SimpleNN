from src.nn import NeuralNet
from src.layers import DenseLayer, InputLayer
from src.utilities import load_data, sigmoid, softplus, softmax, tanh, relu

input_layer = InputLayer((None, 784))
hidden_layer = DenseLayer(input_layer, 30, nonlinearity=softplus)
output_layer = DenseLayer(hidden_layer, 10, nonlinearity=softmax)
layers = [input_layer, hidden_layer, output_layer]

_mini_batch_size = 100
_lr = 0.01
_n_epochs = 15

nn = NeuralNet(layers,
               batch_size=_mini_batch_size,
               regression=False,
               learning_rate=_lr)

train, valid, test = load_data('mnist.pkl.gz', True)

nn.describe()

nn.fit(train, valid, max_iters=_n_epochs)
