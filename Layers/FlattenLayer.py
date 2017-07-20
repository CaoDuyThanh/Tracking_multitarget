import theano.tensor as T

class FlattenLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save information to its layer
        self.N_dim    = _net.layer_opts['flatten_ndim']

        self.output = _input.flatten(ndim = self.N_dim)