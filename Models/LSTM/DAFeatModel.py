import theano
import theano.tensor as T
from Layers.Net import *
from Layers.LayerHelper import *

class DAFeatModel():
    def __init__(self,
                 _dafeat_en_input_size  = 256,
                 _dafeat_en_hidden_size = 512,
                 _dafeat_de_input_size  = 256,
                 _dafeat_de_output_size = 256):
        # ===== Create model =====

        # ----- Create tensor -----
        self.encode_x_batch = T.tensor3('EncodeXBatch')
        self.decode_x_batch = T.tensor3('DecodeXBatch')
        self.decode_y_batch = T.tensor3('DecodeYBatch')

        # ----- Extract Info -----
        _num_truncate = self.encode_x_batch.shape[0]
        _num_object   = self.encode_x_batch.shape[1]

        # ----- Create encoder RNN layer -----
        self.DAFeat_en_net          = RNNNet()
        self.DAFeat_en_net.net_name = 'DAFeat_Encode'
        self.DAFeat_en_net.layer_opts['rnn_input_size']  = _dafeat_en_input_size
        self.DAFeat_en_net.layer_opts['rnn_hidden_size'] = _dafeat_en_hidden_size
        self.DAFeat_en_net.layer_opts['rnn_output_size'] = 2
        self.DAFeat_en_net.layer['rnn_trun']             = RNNLayer(self.DAFeat_en_net)

        _en_output, _en_update = theano.scan(self.DAFeat_en_net.layer['rnn_trun'].step,
                                             sequences    = [self.encode_x_batch],
                                             outputs_info = [T.alloc(numpy_floatX(0.),
                                                                     _num_object,
                                                                     _dafeat_en_hidden_size),
                                                             T.alloc(numpy_floatX(0.),
                                                                     _num_object,
                                                                     2)],
                                             n_steps      = _num_truncate)
        _en_Hs = _en_output[0]

        # ----- Create decoder LSTM layer -----
        self.DAFeat_de_net          = LSTMNet()
        self.DAFeat_de_net.net_name = 'DAFeat_Decode'
        self.DAFeat_de_net.layer_opts['lstm_input_size']  = _dafeat_de_input_size
        self.DAFeat_de_net.layer_opts['lstm_hidden_size'] = _dafeat_en_hidden_size
        self.DAFeat_de_net.layer_opts['lstm_output_size'] = _dafeat_de_output_size
        self.DAFeat_de_net.layer['lstm_trun']             = LSTMLayer(self.DAFeat_de_net)

        _de_output, _de_update = theano.scan(self.DAFeat_de_net.layer['lstm_trun'].step,
                                             sequences    = [self.decode_x_batch,
                                                             _en_Hs],
                                             outputs_info = [T.alloc(numpy_floatX(0.),
                                                                     _num_object,
                                                                     _dafeat_en_hidden_size),
                                                             T.alloc(numpy_floatX(0.),
                                                                     _num_object,
                                                                     _dafeat_de_output_size)],
                                             n_steps      = _num_truncate)
        _prob = _de_output[1]

        # ----- Confidence loss -----
        _A_pred      = T.argmax(_prob, axis = 2)
        _cost_batch  = T.sum(self.decode_y_batch * -T.log(_prob))
        _cost_batch /= _num_truncate

        # ----- Params -----
        _params = self.DAFeat_en_net.layer['rnn_trun'].params[:3] + \
                  self.DAFeat_de_net.layer['lstm_trun'].params
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
                                                     self.encode_x_batch,
                                                     self.decode_x_batch,
                                                     self.decode_y_batch],
                                          updates = updates,
                                          outputs = [_cost_batch, _magnitude])

        # ----- Valid function -----
        self.valid_func = theano.function(inputs  = [self.encode_x_batch,
                                                     self.decode_x_batch,
                                                     self.decode_y_batch],
                                          outputs = [_cost_batch])

    # def create_func(self,
    #                 id):
    #     self.test = theano.function(inputs=[self.encode_x_batch,
    #                                         self.decode_x_batch,
    #                                         self.decode_y_batch],
    #                                 outputs = self.grads[id])

    # def _cost_step(self,
    #                _prob,
    #                _target):
    #     _all_conf_pos_cost = - _target * T.log(_prob)
    #     _all_conf_neg_cost = - (1 - _target) * T.log(1 - _prob)
    #
    #     _all_pos_cost_sum  = T.sum(_all_conf_pos_cost, axis = 1)
    #     _all_neg_cost_sum  = T.sum(_all_conf_neg_cost, axis = 1)
    #
    #     _sorted_pos_cost_idx = T.argsort(_all_pos_cost_sum, axis = 0)
    #     _sorted_neg_cost_idx = T.argsort(_all_neg_cost_sum, axis = 0)
    #
    #     _sorted_pos_cost = _all_pos_cost_sum[_sorted_pos_cost_idx]
    #     _sorted_neg_cost = _all_neg_cost_sum[_sorted_neg_cost_idx]
    #
    #     _num_pos_max = T.sum(T.neg(_sorted_pos_cost, 0))
    #     _num_neg_max = T.cast(T.floor(_num_pos_max * 3), dtype = 'int32')
    #
    #     _top2_pos_cost = _sorted_pos_cost[ : _num_pos_max]
    #     _top6_pos_cost = _sorted_neg_cost[ : _num_neg_max]
    #
    #     _layer_cost = T.where(_num_pos_max > 0, (T.sum(_top2_pos_cost) + T.sum(_top6_pos_cost)) / _num_pos_max, 0)
    #     _layer_pos  = T.where(_num_pos_max > 0, _prob[_sorted_pos_cost_idx[ - _num_pos_max : ]].sum() / _num_pos_max, 0)
    #     _layer_neg  = T.where(_num_pos_max > 0, _prob[_sorted_neg_cost_idx[ - _num_neg_max : ]].sum() / _num_neg_max, 0)
    #
    #     return _layer_cost, _layer_pos, _layer_neg

    def save_model(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].save_model(file)
        self.DAFeat_de_net.layer['lstm_trun'].save_model(file)

    def save_state(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].save_model(file)
        self.DAFeat_de_net.layer['lstm_trun'].save_model(file)
        self.optimizer.save_model(file)

    def load_model(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].load_model(file)
        self.DAFeat_de_net.layer['lstm_trun'].load_model(file)

    def load_state(self, file):
        self.DAFeat_en_net.layer['rnn_trun'].load_model(file)
        self.DAFeat_de_net.layer['lstm_trun'].load_model(file)
        self.optimizer.load_model(file)
