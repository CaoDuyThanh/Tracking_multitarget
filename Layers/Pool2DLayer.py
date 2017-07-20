from theano.tensor.signal.pool import pool_2d

class Pool2DLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save config information to its layer
        self.Ws            = _net.layer_opts['pool_filter_size']
        self.ignore_border = _net.layer_opts['pool_ignore_border']
        self.stride        = _net.layer_opts['pool_stride']
        self.padding       = _net.layer_opts['pool_padding']
        self.mode          = _net.layer_opts['pool_mode']

        self.output = pool_2d(input         = _input,
                              ws            = self.Ws,
                              ignore_border = self.ignore_border,
                              stride        = self.stride,
                              pad           = self.padding,
                              mode          = self.mode)