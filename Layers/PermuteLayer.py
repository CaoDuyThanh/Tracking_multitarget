import theano

class PermuteLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save information to its layer
        self.permute_dimension = _net.layer_opts['permute_dimension']

        self.output = _input.dimshuffle(self.permute_dimension)