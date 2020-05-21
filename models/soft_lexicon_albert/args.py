#coding=utf-8

import os
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

file_path = os.path.dirname(__file__)


#模型目录
model_dir = '/root/zhaomeng/albert_test/pretrain_models/albert_tiny_489k'
#model_dir = '/root/zhaomeng/albert_test/pretrain_models/albert_base_zh_additional_36k_steps'

#config文件
bert_config_file = os.path.join(model_dir, 'albert_config.json')
#ckpt文件名称
ckpt_name = os.path.join(model_dir, 'albert_model.ckpt')
#输出文件目录
output_dir = os.path.join(file_path, 'output/checkpoints')
#vocab文件目录
vocab_file = os.path.join(model_dir, 'vocab.txt')
#数据目录
data_dir = os.path.join(file_path, 'data/')

num_train_epochs = 10
batch_size = 64
learning_rate = 0.00005

# gpu使用率
gpu_memory_fraction = 0.8

# 默认取倒数第二层的输出值作为句向量
layer_indexes = [-2]

# 序列的最大程度，单文本建议把该值调小
max_seq_length = 64

# graph名字
graph_file = os.path.join(model_dir, 'graph')
