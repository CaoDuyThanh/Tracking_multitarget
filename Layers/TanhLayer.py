import theano.tensor as T

class TanhLayer():
    def __init__(self,
                 _input):
        # Save information to its layer
        self.input  = _input

        self.output = T.tanh(_input)