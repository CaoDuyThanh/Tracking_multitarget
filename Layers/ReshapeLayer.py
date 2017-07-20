import theano.tensor as T

class ReshapeLayer():
    def __init__(self,
                 _net,
                 _input):
        # Save all information
        self.new_shape = _net.layer_opts['reshape_new_shape']

        _shape_input = _input.shape

        _new_shape = []
        _pos = -1
        for _idx, _shape in enumerate(self.new_shape):
            if _shape == -1:
                _new_shape += 1
                _pos = _idx
            else:
                if _shape == 0:
                    _new_shape.append(_shape_input[_idx])
                else:
                    _new_shape.append(_shape)
        if _pos >= 0:
            _new_shape[_pos] = T.prod(_shape_input) / T.prod(_new_shape)

        self.output = _input.reshape(tuple(_new_shape))