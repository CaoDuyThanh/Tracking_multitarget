import theano.tensor as T

class CrossEntropyLayer():
    def __init__(self,
                 _pred,
                 _target):
        self.out = - _target * T.log(_pred) - (1 - _target) * T.log(1 - _pred)