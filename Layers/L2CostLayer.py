import theano.tensor as T

class L2CostLayer():
    def __init__(self,
                 _net,
                 _pred,
                 _target):
        # Save all information to its layer
        self.axis = _net.layer_opts['l2cost_axis']

        _out = T.mean(T.sqr(_target - _pred), axis = self.axis, keepdims = True)
        self.output = T.mean(_out)