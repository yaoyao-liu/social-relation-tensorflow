import tensorflow as tf
import numpy as np

slim = tf.contrib.slim

class VGG16:
    def __init__(self):
        self.vgg_mean = [103.939, 116.779, 123.68]

    def forward_network(self, input_, scope="vgg16", reuse=False):
        with tf.variable_scope(scope, reuse=reuse) as vs:
            net = slim.repeat(input_, 2, slim.conv2d, 64, [3, 3], scope='conv1')
            net = slim.max_pool2d(net, [2, 2], scope='pool1')
            net = slim.repeat(net, 2, slim.conv2d, 128, [3, 3], scope='conv2')
            net = slim.max_pool2d(net, [2, 2], scope='pool2')
            net = slim.repeat(net, 3, slim.conv2d, 256, [3, 3], scope='conv3')
            net = slim.max_pool2d(net, [2, 2], scope='pool3')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv4')
            net = slim.max_pool2d(net, [2, 2], scope='pool4')
            net = slim.repeat(net, 3, slim.conv2d, 512, [3, 3], scope='conv5')
            net = slim.max_pool2d(net, [2, 2], scope='pool5')
            output_ = tf.reshape(net, [-1, net.get_shape().as_list()[1]*net.get_shape().as_list()[2]*net.get_shape().as_list()[3]], name='reshape')

        variables = tf.contrib.framework.get_variables(vs)
        return output_, variables
