#coding:utf-8
###################################################
# File Name: highway.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年04月20日 星期五 14时11分50秒
#=============================================================

def highway(input_, num_outputs, num_layers=1, bias=-2.0, f=tf.nn.relu, scope='Highway'):
    '''
    Highway network (cf. http://arxiv.org/abs/1505.00387).
    t = sigmoid(Wy + b)
    z = t * g(Wy + b) + (1 - t) * y
    where g is nonlinearity, t is transform gate, and (1 - t) is carry gate
    '''
    size = int(num_outputs)
    with tf.variable_scope(scope):
        for idx in range(num_layers):
            g = f(tf.contrib.layers.linear(input_, size, scope='highway_lin_%d' % idx))

            t = tf.sigmoid(tf.contrib.layers.linear(input_, size, scope='highway_gate_%d' % idx) + bias)

            output = t * g + (1 - t) * input_
            input_ = output

    return output
