import numpy
import theano
import theano.tensor as T
import caffe
from theano.tensor.shared_randomstreams import RandomStreams

class NeuralNet():
    def __init__(self):
        # Setting default options for NetOpts
        self.net_opts = {}
        self.net_opts['net_state']        = T.scalar('NetState')         # 1: Train | 0: Valid | 0: Test
        self.net_opts['rng_seed']         = 1610
        self.net_opts['rng']              = numpy.random.RandomState(self.net_opts['rng_seed'])
        self.net_opts['theano_rng']       = RandomStreams(self.net_opts['rng'].randint(2 ** 30))
        self.net_opts['learning_rate']    = T.fscalar('LearningRate')
        self.net_opts['batch_size']       = 1

        self.layer_opts = {}
        # Default options for softmax layers
        self.layer_opts['softmax_axis'] = 1

        # Default options for relu layers
        self.layer_opts['relu_alpha'] = 0.01

        # Deafult options for elu layers
        self.layer_opts['elu_alpha'] = 1

        # Default l2 term
        self.layer_opts['l2_term'] = 0.0005

        # Default l2 cost layer
        self.layer_opts['l2cost_axis'] = 1

        # Default dropping rate for dropout
        self.layer_opts['drop_rate']       = 0.5
        self.layer_opts['drop_shape']      = None

        # Default options for hidden layer
        self.layer_opts['hidden_input_size']  = 100
        self.layer_opts['hidden_output_size'] = 100
        self.layer_opts['hidden_W']           = None
        self.layer_opts['hidden_WName']       = ''
        self.layer_opts['hidden_b']           = None
        self.layer_opts['hidden_bName']       = ''

        # Default options for reshape layer
        self.layer_opts['reshape_new_shape']  = None

        # Permute layer
        self.layer_opts['permute_dimension']  = None

        # Flatten layer
        self.layer_opts['flatten_ndim'] = 2

        # Normalize layer
        self.layer_opts['normalize_scale']        = 1
        self.layer_opts['normalize_filter_shape'] = 1
        self.layer_opts['normalize_'] = 1

        # Concatenate layer
        self.layer_opts['concatenate_axis'] = 1

        self.update_opts = {}
        # Adam update
        self.update_opts['adam_beta1'] = 0.9
        self.update_opts['adam_beta2'] = 0.999
        self.update_opts['adam_delta'] = 1e-08


        # Network name for saving
        self.net_name = 'SimpleNet'

        # The content dictionary will store actual layers (LayerHelper)
        self.layer  = {}
        self.Params = []

    def load_caffe_model(self,
                         _caffe_prototxt_path,
                         _caffe_model_path):
        _net_caffe = caffe.Net(_caffe_prototxt_path, _caffe_model_path, caffe.TEST)
        _layers_caffe = dict(zip(list(_net_caffe._layer_names), _net_caffe.layers))

        for _name, _layer in self.layer.items():
            try:
                if _name not in _layers_caffe:
                    continue
                if _name == 'conv4_3_norm':
                    _layer.scale.set_value(_layers_caffe[_name].blobs[0].data)
                _layer.W.set_value(_layers_caffe[_name].blobs[0].data)
                _layer.b.set_value(_layers_caffe[_name].blobs[1].data)
            except AttributeError:
                continue

class ConvNeuralNet(NeuralNet):
    def __init__(self):
        NeuralNet.__init__(self)

        # Setting default options for layer_opts
        # Default options for conv layers
        self.layer_opts['conv2D_input_shape']     = None
        self.layer_opts['conv2D_filter_shape']    = (32, 3, 3, 3)
        self.layer_opts['conv2D_W']               = None
        self.layer_opts['conv2D_WName']           = ''
        self.layer_opts['conv2D_b']               = None
        self.layer_opts['conv2D_bName']           = ''
        self.layer_opts['conv2D_border_mode']     = 'valid'
        self.layer_opts['conv2D_stride']          = (1, 1)
        self.layer_opts['conv2D_filter_flip']     = False
        self.layer_opts['conv2D_filter_dilation'] = (1, 1)

        # Default options for pooling layers
        self.layer_opts['pool_stride']        = (2, 2)
        self.layer_opts['pool_padding']       = (0, 0)
        self.layer_opts['pool_mode']          = 'max'
        self.layer_opts['pool_filter_size']   = (2, 2)
        self.layer_opts['pool_ignore_border'] = False

        # Network name for saving
        self.NetName = 'ConvNet'

class LSTMNet(NeuralNet):
    def __init__(self):
        NeuralNet.__init__(self)

        # Setting default options for layer_opts
        # Default options for lstm layers
        self.layer_opts['lstm_input_size']   = None
        self.layer_opts['lstm_hidden_size']  = 500
        self.layer_opts['lstm_output_size']  = None
        self.layer_opts['lstm_num_truncate'] = 20
        self.layer_opts['lstm_params']       = None

        # Network name for saving
        self.NetName = 'LSTMNet'

class RNNNet(NeuralNet):
    def __init__(self):
        NeuralNet.__init__(self)

        # Setting default options for layer_opts
        # Default options for lstm layers
        self.layer_opts['rnn_input_size']   = None
        self.layer_opts['rnn_hidden_size']  = 500
        self.layer_opts['rnn_output_size']  = 500
        self.layer_opts['rnn_num_truncate'] = 20
        self.layer_opts['rnn_params']       = None

        # Network name for saving
        self.net_name = 'RNNNet'

