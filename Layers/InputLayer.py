
class InputLayer():
    def __init__(self,
                 net,
                 _input):
        # Save all information to its layer
        self.input = _input

        self.output = self.input