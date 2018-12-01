import tensorflow as tf
import numpy as np
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.python.platform import flags

FLAGS = flags.FLAGS
slim = tf.contrib.slim

class RESNET101:
    def __init__(self):
        self.resnet_mean = [0.0, 0.0, 0.0]
        self.is_training = FLAGS.train

    def forward_network(self, input_, scope="resnet101", reuse=False):
        with tf.variable_scope(scope, reuse=reuse) as vs:
            _, end_points = resnet_v1.resnet_v1_101(input_, 1000, is_training=self.is_training)
            import pdb
            pdb.set_trace()
            net = end_points[scope + '/resnet_v1_101/block4']
            output_ = tf.reshape(net0, [-1, net.get_shape().as_list()[1]*net.get_shape().as_list()[2]*net.get_shape().as_list()[3]], name='reshape')

        variables = tf.contrib.framework.get_variables(vs)
        return output_, variables
