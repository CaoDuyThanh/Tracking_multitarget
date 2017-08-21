import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class DAFeatModel():
    def __init__(self,
                 _dafeat_en_input_size  = 256,
                 _dafeat_en_hidden_size = 512):
        # ===== Create model =====

        # ----- Create tensor -----
        self.encode_x_pos_batch = T.tensor3('EncodeXPosBatch')
        self.encode_h_pos_batch = T.matrix('EncodeHPosBatch', dtype ='float32')
        self.decode_x_batch     = T.tensor4('DecodeXBatch')
        self.decode_y_batch     = T.tensor4('DecodeYBatch')

        # ----- Extract Info -----
        _num_truncate = self.encode_x_pos_batch.shape[0]
        _num_object   = self.encode_x_pos_batch.shape[1]

        # ----- Create encoder RNN layer -----
        self.DAFeat_en_net          = RNNNet()
        self.DAFeat_en_net.net_name = 'DAFeat_Encode'
        self.DAFeat_en_net.layer_opts['rnn_input_size']  = _dafeat_en_input_size
        self.DAFeat_en_net.layer_opts['rnn_hidden_size'] = _dafeat_en_hidden_size
        self.DAFeat_en_net.layer_opts['rnn_output_size'] = 2
        self.DAFeat_en_net.layer['rnn_trun']             = RNNLayer(self.DAFeat_en_net)

        _en_output, _en_update = theano.scan(self.DAFeat_en_net.layer['rnn_trun'].step,
                                             sequences    = [self.encode_x_pos_batch],
                                             outputs_info = [self.encode_h_pos_batch,
                                                             None],
                                             n_steps      = _num_truncate)
        _en_Hs     = _en_output[0]
        _last_en_H = _en_Hs[-1, ]

        # ----- Create decoder LSTM layer -----
        self.DAFeat_de_net = RNNNet()
        self.DAFeat_de_net.net_name = 'DAFeat_Decode'
        self.DAFeat_de_net.layer_opts['rnn_input_size']  = _dafeat_en_input_size
        self.DAFeat_de_net.layer_opts['rnn_hidden_size'] = _dafeat_en_hidden_size
        self.DAFeat_de_net.layer_opts['rnn_output_size'] = 1
        self.DAFeat_de_net.layer['rnn_trun'] = RNNLayer(self.DAFeat_de_net)
        _de_output, _de_update = theano.scan(self.DAFeat_de_net.layer['rnn_trun'].step1,
                                             sequences    = [self.decode_x_batch,
                                                             _en_Hs],
                                             outputs_info = [None],
                                             n_steps      = _num_truncate)
        _output = _de_output
        _prob   = T.nnet.sigmoid(_output)

        # ----- Confidence loss -----
        _pred        = T.argmax(_prob, axis = 2)
        _truth       = T.argmax(self.decode_y_batch, axis = 2)
        _precision   = T.mean(T.eq(_pred, _truth))
        _all_cost    = self.decode_y_batch * -T.log(_prob) + (1 - self.decode_y_batch) * -T.log(1 - _prob)
        _cost_batch  = T.mean(_all_cost)

        # ----- Params -----
        _params = self.DAFeat_en_net.layer['rnn_trun'].params[0:3] + \
                  self.DAFeat_de_net.layer['rnn_trun'].params
        _grads  = T.grad(_cost_batch, _params)
        self.grads = _grads

        # ----- Optimizer -----
        self.optimizer = AdamGDUpdate(self.DAFeat_en_net, params = _params, grads = _grads)
        _magnitude = 0
        for grad in self.optimizer.grads:
            _magnitude += T.sqr(grad).sum()
        _magnitude = T.sqrt(_magnitude)

        # ----- Update -----
        updates = _en_update + \
                  _de_update + \
                  self.optimizer.updates

        # ===== Functions =====
        # ----- Train function -----
        self.train_func = theano.function(inputs  = [self.optimizer.learning_rate,
                                                     self.encode_x_pos_batch,
                                                     self.encode_h_pos_batch,
                                                     self.decode_x_batch,
                                                     self.decode_y_batch],
                                          updates = updates,
                                          outputs = [_cost_batch,
                                                     _last_en_H,
                                                     _magnitude,
                                                     _prob,
                                                     self.decode_y_batch])

        # ----- Valid function -----
        self.valid_func = theano.function(inputs  = [self.encode_x_pos_batch,
                                                     self.encode_h_pos_batch,
                                                     self.decode_x_batch,
                                                     self.decode_y_batch],
                                          outputs = [_cost_batch,
                                                     _precision,
                                                     _last_en_H])

        # ----- Valid function -----
        self.cost_func = theano.function(inputs=[self.encode_x_pos_batch,
                                                 self.encode_h_pos_batch,
                                                 self.decode_x_batch],
                                         outputs=[_prob,
                                                  _last_en_H])

        # ----- Pred function -----
        self.pred_func = theano.function(inputs  = [self.encode_x_pos_batch,
                                                    self.encode_h_pos_batch,
                                                    self.decode_x_batch],
                                         outputs = [_pred,
                                                    _last_en_H,
                                                    _prob])

    def save_model(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].save_model(file)
        self.DAFeat_de_net.layer['rnn_trun'].save_model(file)

    def save_state(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].save_model(file)
        self.DAFeat_de_net.layer['rnn_trun'].save_model(file)
        self.optimizer.save_model(file)

    def load_model(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].load_model(file)
        self.DAFeat_de_net.layer['rnn_trun'].load_model(file)

    def load_state(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].load_model(file)
        self.DAFeat_de_net.layer['rnn_trun'].load_model(file)
        self.optimizer.load_model(file)
