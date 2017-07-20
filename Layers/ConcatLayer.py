import theano.tensor as T

class ConcatLayer():
    def __init__(self,
                 _net,
                 _inputs):
        # Save all information to its layer
        self.axis = _net.layer_opts['concatenate_axis']

        self.output = T.concatenate(tuple(_inputs), axis = self.axis)