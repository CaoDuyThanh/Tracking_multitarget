import theano
import theano.tensor as T
import numpy
from BasicLayer import *

class AdamGDUpdate(BasicLayer):
    def __init__(self,
                 net,
                 params,
                 grads):
        # Save all information to its layer
        self.beta1         = net.update_opts['adam_beta1']
        self.beta2         = net.update_opts['adam_beta2']
        self.delta         = net.update_opts['adam_delta']
        self.learning_rate = net.net_opts['learning_rate']

        i = theano.shared(numpy.float32(0.))
        i_t = i + 1

        fix1 = 1. - (1. - self.beta1) ** i_t
        fix2 = 1. - (1. - self.beta2) ** i_t
        lr_t = self.learning_rate * (T.sqrt(fix2) / fix1)

        self.params = [i]
        self.grads  = []
        updates     = []
        for (param, grad) in zip(params, grads):
            mt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            vt = theano.shared(param.get_value() * 0., broadcastable=param.broadcastable)
            self.params.append(mt)
            self.params.append(vt)

            newMt = self.beta1 * mt + (1 - self.beta1) * grad
            newVt = self.beta2 * vt + (1 - self.beta2) * T.sqr(grad)

            step = - lr_t * newMt / (T.sqrt(newVt) + self.delta)
            self.grads.append(step)

            updates.append((mt, newMt))
            updates.append((vt, newVt))
            updates.append((param, param + step))
        updates.append((i, i_t))

        self.updates = updates

