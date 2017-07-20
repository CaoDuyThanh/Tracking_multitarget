import numpy
import theano

def numpy_floatX(data):
    return numpy.asarray(data, dtype = 'float32')

def create_shared_parameter(_rng      = None,
                            _shape    = None,
                            _W_bound  = None,
                            _factor   = 1,
                            _name_var =''):
    if _rng is None:
        _rng = numpy.random.RandomState(1993)

    if _W_bound is None:
        _W_bound = numpy.sqrt(6.0 / numpy.sum(_shape))
    _init_value = numpy.asarray(_rng.uniform(
                                    low   = -_W_bound,
                                    high  =  _W_bound,
                                    size  =  _shape
                                ),
                                dtype = theano.config.floatX
                            )
    _shared_var = theano.shared(_init_value * _factor, borrow = True, name = _name_var)
    return _shared_var

def create_ortho_parameter(_rng      = None,
                           _shape    = None,
                           _factor   = 1,
                           _name_var =''):
    if _rng is None:
        _rng = numpy.random.RandomState(1993)

    _init_value = numpy.asarray(_rng.normal(size = _shape),
                                dtype = theano.config.floatX)
    _shared_var = theano.shared(_init_value * _factor, borrow = True, name = _name_var)
    return _shared_var