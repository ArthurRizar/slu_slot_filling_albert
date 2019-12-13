#coding:utf-8
###################################################
# File Name: embedding.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年04月16日 星期一 17时00分27秒
#=============================================================

import tensorflow as tf


class Embedding(object):
    def __init__(self, vocab_size, embed_dim, position_size=None, embedding_table=None, use_subword=False, trainable=True, POS_onehot_size=0):
        self.use_subword = use_subword
        self.trainable = trainable
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.position_size = position_size
        self.POS_onehot_size = POS_onehot_size

        with tf.device('/cpu:0'):
            if embedding_table is None:
                self.W = tf.Variable(tf.random_uniform([vocab_size, self.embed_dim], -1, 1), name='W', trainable=self.trainable)
            else:
                self.W = tf.Variable(embedding_table, name='W', trainable=self.trainable)

    def get_embedded_inputs(self, input_x, input_POS=None, input_sparse_x=None):
        if self.use_subword:
            self.embedded_word = tf.nn.embedding_lookup_sparse(self.W, input_sparse_x, None, combiner='mean')
            self.embedded_word = tf.reshape(self.embedded_word, [-1, seq_len, self.embed_dim])
        else:
            self.embedded_word = tf.nn.embedding_lookup(self.W, input_x)
        final_embedding_list = [self.embedded_word]

        if self.POS_onehot_size is not None and input_POS is not None:
            POS_onehot = tf.one_hot(input_POS, self.POS_onehot_size, 1.0, 0.0)
            final_embedding_list.append(POS_onehot) 
        if self.position_size is not None:
            position_embedding = self.calc_position_embedding(input_x, self.position_size)
            final_embedding_list.append(position_embedding)
        
        self.embedded_word_POS = tf.concat(final_embedding_list, axis=2)
        return self.embedded_word_POS

    def calc_position_embedding(self, inputs, position_size):
        batch_size,seq_len = tf.shape(inputs)[0],tf.shape(inputs)[1]
        position_j = 1. / tf.pow(10000., \
                             2 * tf.range(position_size / 2, dtype=tf.float32 \
                            ) / position_size)
        position_j = tf.expand_dims(position_j, 0)
        position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
        position_i = tf.expand_dims(position_i, 1)
        position_ij = tf.matmul(position_i, position_j)
        position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
        position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
        return position_embedding 



if __name__ == '__main__':
    pass
