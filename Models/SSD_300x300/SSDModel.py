import theano
import theano.tensor as T
import numpy
import cv2
from Layers.LayerHelper import *
from Layers.Net import *

class SSDModel():
    def __init__(self):
        ####################################
        #       Create model               #
        ####################################

        # Create tensor variables to store input / output data
        self.X = T.tensor4('X')

        # Create shared variable for input
        net = ConvNeuralNet()
        net.net_name = 'SSD Net'

        _batch_size = self.X.shape[0]

        # Input
        net.layer['input_4d'] = InputLayer(net, self.X)

        net.layer_opts['pool_boder_mode']    = 1
        net.layer_opts['conv2D_border_mode'] = 1

        # Stack 1
        net.layer_opts['conv2D_filter_shape'] = (64, 3, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv1_1_W'
        net.layer_opts['conv2D_bName'] = 'conv1_1_b'
        net.layer['conv1_1'] = ConvLayer(net, net.layer['input_4d'].output)
        net.layer['relu1_1'] = ReLULayer(net.layer['conv1_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (64, 64, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv1_2_W'
        net.layer_opts['conv2D_bName'] = 'conv1_2_b'
        net.layer['conv1_2'] = ConvLayer(net, net.layer['relu1_1'].output)
        net.layer['relu1_2'] = ReLULayer(net.layer['conv1_2'].output)

        net.layer_opts['pool_mode'] = 'max'
        net.layer['pool1']   = Pool2DLayer(net, net.layer['relu1_2'].output)

        # Stack 2
        net.layer_opts['conv2D_filter_shape'] = (128, 64, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv2_1_W'
        net.layer_opts['conv2D_bName'] = 'conv2_1_b'
        net.layer['conv2_1'] = ConvLayer(net, net.layer['pool1'].output)
        net.layer['relu2_1'] = ReLULayer(net.layer['conv2_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (128, 128, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv2_2_W'
        net.layer_opts['conv2D_bName'] = 'conv2_2_b'
        net.layer['conv2_2'] = ConvLayer(net, net.layer['relu2_1'].output)
        net.layer['relu2_2'] = ReLULayer(net.layer['conv2_2'].output)

        net.layer['pool2']   = Pool2DLayer(net, net.layer['relu2_2'].output)

        # Stack 3
        net.layer_opts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv3_1_W'
        net.layer_opts['conv2D_bName'] = 'conv3_1_b'
        net.layer['conv3_1'] = ConvLayer(net, net.layer['pool2'].output)
        net.layer['relu3_1'] = ReLULayer(net.layer['conv3_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (256, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv3_2_W'
        net.layer_opts['conv2D_bName'] = 'conv3_2_b'
        net.layer['conv3_2'] = ConvLayer(net, net.layer['relu3_1'].output)
        net.layer['relu3_2'] = ReLULayer(net.layer['conv3_2'].output)

        net.layer_opts['conv2D_filter_shape'] = (256, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv3_3_W'
        net.layer_opts['conv2D_bName'] = 'conv3_3_b'
        net.layer['conv3_3'] = ConvLayer(net, net.layer['relu3_2'].output)
        net.layer['relu3_3'] = ReLULayer(net.layer['conv3_3'].output)

        net.layer['pool3']   = Pool2DLayer(net, net.layer['relu3_3'].output)

        # Stack 4
        net.layer_opts['conv2D_filter_shape'] = (512, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv4_1_W'
        net.layer_opts['conv2D_bName'] = 'conv4_1_b'
        net.layer['conv4_1'] = ConvLayer(net, net.layer['pool3'].output)
        net.layer['relu4_1'] = ReLULayer(net.layer['conv4_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv4_2_W'
        net.layer_opts['conv2D_bName'] = 'conv4_2_b'
        net.layer['conv4_2'] = ConvLayer(net, net.layer['relu4_1'].output)
        net.layer['relu4_2'] = ReLULayer(net.layer['conv4_2'].output)

        net.layer_opts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv4_3_W'
        net.layer_opts['conv2D_bName'] = 'conv4_3_b'
        net.layer['conv4_3'] = ConvLayer(net, net.layer['relu4_2'].output)
        net.layer['relu4_3'] = ReLULayer(net.layer['conv4_3'].output)

        net.layer['pool4']   = Pool2DLayer(net, net.layer['relu4_3'].output)
        net.layer_opts['normalize_scale']        = 20
        net.layer_opts['normalize_filter_shape'] = (512, )
        net.layer_opts['normalize_scale_name']   = 'conv4_3_scale'
        net.layer['conv4_3_norm']        = NormalizeLayer(net, net.layer['relu4_3'].output)

        # conv4_3_norm_mbox_conf
        net.layer_opts['conv2D_filter_shape'] = (84, 512, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv4_3_norm_mbox_conf_W'
        net.layer_opts['conv2D_bName'] = 'conv4_3_norm_mbox_conf_b'
        net.layer['conv4_3_norm_mbox_conf'] = ConvLayer(net, net.layer['conv4_3_norm'].output)

        net.layer_opts['permute_dimension']       = (0, 2, 3, 1)
        net.layer['conv4_3_norm_mbox_conf_perm'] = PermuteLayer(net, net.layer['conv4_3_norm_mbox_conf'].output)
        net.layer_opts['flatten_ndim']            = 2
        net.layer['conv4_3_norm_mbox_conf_flat'] = FlattenLayer(net, net.layer['conv4_3_norm_mbox_conf_perm'].output)

        # conv4_3_norm_mbox_loc
        net.layer_opts['conv2D_filter_shape'] = (16, 512, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv4_3_norm_mbox_loc_W'
        net.layer_opts['conv2D_bName'] = 'conv4_3_norm_mbox_loc_b'
        net.layer['conv4_3_norm_mbox_loc'] = ConvLayer(net, net.layer['conv4_3_norm'].output)

        net.layer_opts['permute_dimension']      = (0, 2, 3, 1)
        net.layer['conv4_3_norm_mbox_loc_perm'] = PermuteLayer(net, net.layer['conv4_3_norm_mbox_loc'].output)
        net.layer_opts['flatten_ndim']           = 2
        net.layer['conv4_3_norm_mbox_loc_flat'] = FlattenLayer(net, net.layer['conv4_3_norm_mbox_loc_perm'].output)

        # Stack 5
        net.layer_opts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.layer_opts['conv2D_WName'] = 'conv5_1_W'
        net.layer_opts['conv2D_bName'] = 'conv5_1_b'
        net.layer['conv5_1'] = ConvLayer(net, net.layer['pool4'].output)
        net.layer['relu5_1'] = ReLULayer(net.layer['conv5_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.layer_opts['conv2D_WName'] = 'conv5_2_W'
        net.layer_opts['conv2D_bName'] = 'conv5_2_b'
        net.layer['conv5_2'] = ConvLayer(net, net.layer['relu5_1'].output)
        net.layer['relu5_2'] = ReLULayer(net.layer['conv5_2'].output)

        net.layer_opts['conv2D_filter_shape'] = (512, 512, 3, 3)
        net.layer_opts['conv2D_WName'] = 'conv5_3_W'
        net.layer_opts['conv2D_bName'] = 'conv5_3_b'
        net.layer['conv5_3'] = ConvLayer(net, net.layer['relu5_2'].output)
        net.layer['relu5_3'] = ReLULayer(net.layer['conv5_3'].output)

        net.layer_opts['pool_ignore_border'] = True
        net.layer_opts['pool_filter_size']   = (3, 3)
        net.layer_opts['pool_stride']        = (1, 1)
        net.layer_opts['pool_padding']       = (1, 1)
        net.layer['pool5']    = Pool2DLayer(net, net.layer['relu5_3'].output)

        # fc6 and fc7
        net.layer_opts['conv2D_filter_shape']    = (1024, 512, 3, 3)
        net.layer_opts['conv2D_stride']          = (1, 1)
        net.layer_opts['conv2D_border_mode']     = (6, 6)
        net.layer_opts['conv2D_filter_dilation'] = (6, 6)
        net.layer_opts['conv2D_WName'] = 'fc6_W'
        net.layer_opts['conv2D_bName'] = 'fc6_b'
        net.layer['fc6']   = ConvLayer(net, net.layer['pool5'].output)
        net.layer['relu6'] = ReLULayer(net.layer['fc6'].output)
        net.layer_opts['conv2D_filter_dilation'] = (1, 1)        # Set default filter dilation

        net.layer_opts['conv2D_filter_shape'] = (1024, 1024, 1, 1)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = 0
        net.layer_opts['conv2D_WName']        = 'fc7_W'
        net.layer_opts['conv2D_bName']        = 'fc7_b'
        net.layer['fc7']   = ConvLayer(net, net.layer['relu6'].output)
        net.layer['relu7'] = ReLULayer(net.layer['fc7'].output)

        # First sub convolution to get predicted box
        # fc7_mbox_conf
        net.layer_opts['conv2D_filter_shape'] = (126, 1024, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'fc7_mbox_conf_W'
        net.layer_opts['conv2D_bName']        = 'fc7_mbox_conf_b'
        net.layer['fc7_mbox_conf']  = ConvLayer(net, net.layer['relu7'].output)

        net.layer_opts['permute_dimension'] = (0, 2, 3, 1)
        net.layer['fc7_mbox_conf_perm']    = PermuteLayer(net, net.layer['fc7_mbox_conf'].output)
        net.layer_opts['flatten_ndim']      = 2
        net.layer['fc7_mbox_conf_flat']    = FlattenLayer(net, net.layer['fc7_mbox_conf_perm'].output)

        # conv6_1 and conv6_2
        net.layer_opts['conv2D_filter_shape'] = (256, 1024, 1, 1)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = 0
        net.layer_opts['conv2D_WName']        = 'conv6_1_W'
        net.layer_opts['conv2D_bName']        = 'conv6_1_b'
        net.layer['conv6_1']      = ConvLayer(net, net.layer['relu7'].output)
        net.layer['conv6_1_relu'] = ReLULayer(net.layer['conv6_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (512, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (2, 2)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'conv6_2_W'
        net.layer_opts['conv2D_bName']        = 'conv6_2_b'
        net.layer['conv6_2'] = ConvLayer(net, net.layer['conv6_1_relu'].output)
        net.layer['conv6_2_relu'] = ReLULayer(net.layer['conv6_2'].output)

        # fc7_mbox_loc
        net.layer_opts['conv2D_filter_shape'] = (24, 1024, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'fc7_mbox_loc_W'
        net.layer_opts['conv2D_bName']        = 'fc7_mbox_loc_b'
        net.layer['fc7_mbox_loc'] = ConvLayer(net, net.layer['relu7'].output)

        net.layer_opts['permute_dimension'] = (0, 2, 3, 1)
        net.layer['fc7_mbox_loc_perm']     = PermuteLayer(net, net.layer['fc7_mbox_loc'].output)
        net.layer_opts['flatten_ndim']      = 2
        net.layer['fc7_mbox_loc_flat']     = FlattenLayer(net, net.layer['fc7_mbox_loc_perm'].output)

        # Second sub convolution to get predicted box
        # conv6_2_mbox_conf
        net.layer_opts['conv2D_filter_shape'] = (126, 512, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'conv6_2_mbox_conf_W'
        net.layer_opts['conv2D_bName']        = 'conv6_2_mbox_conf_b'
        net.layer['conv6_2_mbox_conf'] = ConvLayer(net, net.layer['conv6_2_relu'].output)

        net.layer_opts['permute_dimension']  = (0, 2, 3, 1)
        net.layer['conv6_2_mbox_conf_perm'] = PermuteLayer(net, net.layer['conv6_2_mbox_conf'].output)
        net.layer_opts['flatten_ndim']       = 2
        net.layer['conv6_2_mbox_conf_flat'] = FlattenLayer(net, net.layer['conv6_2_mbox_conf_perm'].output)

        # conv7_1 and conv7_2
        net.layer_opts['conv2D_filter_shape'] = (128, 512, 1, 1)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = 0
        net.layer_opts['conv2D_WName']        = 'conv7_1_W'
        net.layer_opts['conv2D_bName']        = 'conv7_1_b'
        net.layer['conv7_1']      = ConvLayer(net, net.layer['conv6_2_relu'].output)
        net.layer['conv7_1_relu'] = ReLULayer(net.layer['conv7_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.layer_opts['conv2D_stride']       = (2, 2)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'conv7_2_W'
        net.layer_opts['conv2D_bName']        = 'conv7_2_b'
        net.layer['conv7_2']      = ConvLayer(net, net.layer['conv7_1_relu'].output)
        net.layer['conv7_2_relu'] = ReLULayer(net.layer['conv7_2'].output)

        # conv6_2_mbox_loc
        net.layer_opts['conv2D_filter_shape'] = (24, 512, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'conv6_2_mbox_loc_W'
        net.layer_opts['conv2D_bName']        = 'conv6_2_mbox_loc_b'
        net.layer['conv6_2_mbox_loc'] = ConvLayer(net, net.layer['conv6_2_relu'].output)

        net.layer_opts['permute_dimension'] = (0, 2, 3, 1)
        net.layer['conv6_2_mbox_loc_perm'] = PermuteLayer(net, net.layer['conv6_2_mbox_loc'].output)
        net.layer_opts['flatten_ndim']      = 2
        net.layer['conv6_2_mbox_loc_flat'] = FlattenLayer(net, net.layer['conv6_2_mbox_loc_perm'].output)

        # Third sub convolution to get predicted box
        # conv7_2_mbox_conf
        net.layer_opts['conv2D_filter_shape'] = (126, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'conv7_2_mbox_conf_W'
        net.layer_opts['conv2D_bName']        = 'conv7_2_mbox_conf_b'
        net.layer['conv7_2_mbox_conf'] = ConvLayer(net, net.layer['conv7_2_relu'].output)

        net.layer_opts['permute_dimension']  = (0, 2, 3, 1)
        net.layer['conv7_2_mbox_conf_perm'] = PermuteLayer(net, net.layer['conv7_2_mbox_conf'].output)
        net.layer_opts['flatten_ndim']       = 2
        net.layer['conv7_2_mbox_conf_flat'] = FlattenLayer(net, net.layer['conv7_2_mbox_conf_perm'].output)

        # conv8_1 and conv8_2
        net.layer_opts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = 0
        net.layer_opts['conv2D_WName']        = 'conv8_1_W'
        net.layer_opts['conv2D_bName']        = 'conv8_1_b'
        net.layer['conv8_1']      = ConvLayer(net, net.layer['conv7_2_relu'].output)
        net.layer['conv8_1_relu'] = ReLULayer(net.layer['conv8_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = 0
        net.layer_opts['conv2D_WName']        = 'conv8_2_W'
        net.layer_opts['conv2D_bName']        = 'conv8_2_b'
        net.layer['conv8_2'] = ConvLayer(net, net.layer['conv8_1_relu'].output)
        net.layer['conv8_2_relu'] = ReLULayer(net.layer['conv8_2'].output)

        # conv7_2_mbox_loc
        net.layer_opts['conv2D_filter_shape'] = (24, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv7_2_mbox_loc_W'
        net.layer_opts['conv2D_bName'] = 'conv7_2_mbox_loc_b'
        net.layer['conv7_2_mbox_loc'] = ConvLayer(net, net.layer['conv7_2_relu'].output)

        net.layer_opts['permute_dimension'] = (0, 2, 3, 1)
        net.layer['conv7_2_mbox_loc_perm'] = PermuteLayer(net, net.layer['conv7_2_mbox_loc'].output)
        net.layer_opts['flatten_ndim']      = 2
        net.layer['conv7_2_mbox_loc_flat'] = FlattenLayer(net, net.layer['conv7_2_mbox_loc_perm'].output)

        # Fourth sub convolution to get predicted box
        # conv8_2_mbox_conf
        net.layer_opts['conv2D_filter_shape'] = (84, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName']        = 'conv8_2_mbox_conf_W'
        net.layer_opts['conv2D_bName']        = 'conv8_2_mbox_conf_b'
        net.layer['conv8_2_mbox_conf'] = ConvLayer(net, net.layer['conv8_2_relu'].output)

        net.layer_opts['permute_dimension']  = (0, 2, 3, 1)
        net.layer['conv8_2_mbox_conf_perm'] = PermuteLayer(net, net.layer['conv8_2_mbox_conf'].output)
        net.layer_opts['flatten_ndim']       = 2
        net.layer['conv8_2_mbox_conf_flat'] = FlattenLayer(net, net.layer['conv8_2_mbox_conf_perm'].output)

        # conv9_1 and conv9_2
        net.layer_opts['conv2D_filter_shape'] = (128, 256, 1, 1)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = 0
        net.layer_opts['conv2D_WName']        = 'conv9_1_W'
        net.layer_opts['conv2D_bName']        = 'conv9_1_b'
        net.layer['conv9_1']      = ConvLayer(net, net.layer['conv8_2_relu'].output)
        net.layer['conv9_1_relu'] = ReLULayer(net.layer['conv9_1'].output)

        net.layer_opts['conv2D_filter_shape'] = (256, 128, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = 0
        net.layer_opts['conv2D_WName'] = 'conv9_2_W'
        net.layer_opts['conv2D_bName'] = 'conv9_2_b'
        net.layer['conv9_2']      = ConvLayer(net, net.layer['conv9_1_relu'].output)
        net.layer['conv9_2_relu'] = ReLULayer(net.layer['conv9_2'].output)

        # conv8_2_mbox_loc
        net.layer_opts['conv2D_filter_shape'] = (16, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv8_2_mbox_loc_W'
        net.layer_opts['conv2D_bName'] = 'conv8_2_mbox_loc_b'
        net.layer['conv8_2_mbox_loc'] = ConvLayer(net, net.layer['conv8_2_relu'].output)

        net.layer_opts['permute_dimension'] = (0, 2, 3, 1)
        net.layer['conv8_2_mbox_loc_perm'] = PermuteLayer(net, net.layer['conv8_2_mbox_loc'].output)
        net.layer_opts['flatten_ndim']      = 2
        net.layer['conv8_2_mbox_loc_flat'] = FlattenLayer(net, net.layer['conv8_2_mbox_loc_perm'].output)

        # Fifth sub convolution to get predicted box
        # conv9_2_mbox_conf
        net.layer_opts['conv2D_filter_shape'] = (84, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv9_2_mbox_conf_W'
        net.layer_opts['conv2D_bName'] = 'conv9_2_mbox_conf_b'
        net.layer['conv9_2_mbox_conf'] = ConvLayer(net, net.layer['conv9_2_relu'].output)

        net.layer_opts['permute_dimension']  = (0, 2, 3, 1)
        net.layer['conv9_2_mbox_conf_perm'] = PermuteLayer(net, net.layer['conv9_2_mbox_conf'].output)
        net.layer_opts['flatten_ndim']       = 2
        net.layer['conv9_2_mbox_conf_flat'] = FlattenLayer(net, net.layer['conv9_2_mbox_conf_perm'].output)

        # conv9_2_mbox_loc
        net.layer_opts['conv2D_filter_shape'] = (16, 256, 3, 3)
        net.layer_opts['conv2D_stride']       = (1, 1)
        net.layer_opts['conv2D_border_mode']  = (1, 1)
        net.layer_opts['conv2D_WName'] = 'conv9_2_mbox_loc_W'
        net.layer_opts['conv2D_bName'] = 'conv9_2_mbox_loc_b'
        net.layer['conv9_2_mbox_loc'] = ConvLayer(net, net.layer['conv9_2_relu'].output)

        net.layer_opts['permute_dimension'] = (0, 2, 3, 1)
        net.layer['conv9_2_mbox_loc_perm'] = PermuteLayer(net, net.layer['conv9_2_mbox_loc'].output)
        net.layer_opts['flatten_ndim']      = 2
        net.layer['conv9_2_mbox_loc_flat'] = FlattenLayer(net, net.layer['conv9_2_mbox_loc_perm'].output)

        # Concat mbox_conf and mbox_loc
        net.layer['mbox_conf'] = ConcatLayer(net, [net.layer['conv4_3_norm_mbox_conf_flat'].output,
                                                   net.layer['fc7_mbox_conf_flat'].output,
                                                   net.layer['conv6_2_mbox_conf_flat'].output,
                                                   net.layer['conv7_2_mbox_conf_flat'].output,
                                                   net.layer['conv8_2_mbox_conf_flat'].output,
                                                   net.layer['conv9_2_mbox_conf_flat'].output])
        net.layer['mbox_loc']  = ConcatLayer(net, [net.layer['conv4_3_norm_mbox_loc_flat'].output,
                                                   net.layer['fc7_mbox_loc_flat'].output,
                                                   net.layer['conv6_2_mbox_loc_flat'].output,
                                                   net.layer['conv7_2_mbox_loc_flat'].output,
                                                   net.layer['conv8_2_mbox_loc_flat'].output,
                                                   net.layer['conv9_2_mbox_loc_flat'].output])

        net.layer_opts['reshape_new_shape'] = (_batch_size, 8732, 21)
        net.layer['mbox_conf_reshape']     = ReshapeLayer(net, net.layer['mbox_conf'].output)

        net.layer_opts['softmax_axis']  = 2
        net.layer['mbox_conf_softmax'] = SoftmaxLayer(net, net.layer['mbox_conf_reshape'].output)

        net.layer_opts['reshape_new_shape'] = (_batch_size, 8732, 4)
        net.layer['mbox_loc_flatten']      = ReshapeLayer(net, net.layer['mbox_loc'].output)

        self.net = net

        # Predict function
        label = T.argmax(net.layer['mbox_conf_softmax'].output, axis = 2, keepdims = True)
        self.pred_func = theano.function(
                            inputs  = [self.X],
                            outputs = [label,
                                       net.layer['mbox_loc_flatten'].output])

        self.test_func = theano.function(
                            inputs  = [self.X],
                            outputs = [net.layer['mbox_conf_softmax'].output])

    def load_caffe_model(self,
                         _caffe_prototxt_path,
                         _caffe_model_path):
        self.net.load_caffe_model(_caffe_prototxt_path, _caffe_model_path)

    def test_network(self,
                     _im):
        return self.pred_func(_im)
