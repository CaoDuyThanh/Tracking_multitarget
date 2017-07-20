import theano.tensor as T

class SigmoidLayer():
    def __init__(self,
                 _input):
        # Save information to its layer
        self.input = _input

        self.output = T.nnet.sigmoid(_input)