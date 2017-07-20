import theano.tensor as T

class ReLULayer():
    def __init__(self,
                 _input):
        # Save information to its layer
        self.input = _input

        self.output = T.nnet.relu(_input)