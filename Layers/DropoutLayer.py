import theano
import theano.tensor as T

class DropoutLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save information config to its layer
        self.state      = _net.net_opts['net_state']
        self.drop_rate  = _net.LayerOpts['drop_rate']
        self.drop_shape = _net.LayerOpts['drop_shape']

        _theano_rng = _net.net_opts['theano_rng']
        self.output = T.switch(self.state,
                               _theano_rng.binomial(size  = self.drop_shape,  # Training state
                                                    n     = 1,
                                                    p     = 1 - self.drop_rate,
                                                    dtype = theano.config.floatX) * _input,
                               _input * (1. - self.drop_rate))                                 # Valid | Test state


