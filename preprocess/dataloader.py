#coding:utf-8
###################################################
# File Name: dataloader.py
# Author: Meng Zhao
# mail: @
# Created Time: Wed 21 Mar 2018 07:04:35 PM CST
#=============================================================
import os
import sys
import time
import datetime
import gensim
import codecs
import numpy as np
import tensorflow as tf
sys.path.append('../')


from tensorflow.contrib import learn
from nltk.util import ngrams
from setting import *
from common.strutil import stringhandler
special_words = set(['<num>', '<phone>'])


def generate_subword(word_uni, max_num):
    '''
    @params: word_uni, unicode
             max_num, the maximum number of subword
    @return suwords, utf8
    '''
    cur_num = 0
    subwords = []
    for i in xrange(2, len(word_uni)):
        subword_iter = ngrams(word_uni, i)
        for subword in subword_iter:
            if cur_num >= max_num:
                break
            subword = ''.join(subword).encode('utf-8')
            subwords.append(subword)
            cur_num += 1
    return subwords, cur_num



def trans_input_expand_subword(x_text, vocab_processor, seq_len, max_num=8):
    vocab_dict = vocab_processor.vocabulary_._mapping
    x = []
    all_nums = []
    for text in x_text:
        text_indices = [] #当前问句的每个word的subwords, list
        cur_nums = [] #当前问句的每个word的subword个数
        words = text.split(' ')
        print(text, 'len:', len(words))
        for word_uni in words:
            subwords, subword_num = generate_subword(word_uni, max_num)
            subwords_str = ' '.join(subwords)
            print(subwords_str)
            word_subword_indices = [vocab_dict[word_uni] if word_uni in vocab_dict else 0]
            subword_indices = [vocab_dict[i.decode('utf-8')] if i.decode('utf-8') in vocab_dict else 0 for i in subwords]
            word_subword_indices.extend(subword_indices)
            word_subword_indices += [0] * (max_num - subword_num) #subword padding
            text_indices.append(word_subword_indices)
            cur_nums.append(subword_num)
        text_indices = text_indices + [[0] * (max_num)] * (seq_len - len(words)) #word padding
        x.append(text_indices)
        all_nums.append(cur_nums)
    return x, all_nums


def get_vocab_idx2word(vocab_dict):
    vocab_idx2word = {}
    for word in vocab_dict:
        idx = vocab_dict[word]
        vocab_idx2word[idx] = word
    return vocab_idx2word


def trans_to_padded_text(x, vocab_dict):
    vocab_idx2word = get_vocab_idx2word(vocab_dict)
    padded_x_text = []
    for text_indices in x:
        padded_text = [vocab_idx2word[idx] for idx in text_indices]
        padded_x_text.append(' '.join(padded_text))
    return padded_x_text


def trans_input_to_sparse(x_text, vocab_processor, seq_len, max_num=8):
    '''
    @breif: prepare data for SparseTensor which has params 'indice', 'values' and 'shape'
            there is:
            sparse_x = tf.sparse_placeholder(tf.int32)
            shape = [seq_len, max_subword_num]
            emb = tf.nn.embedding_lookup_sparse(embedding, sparse_x, None, combiner='mean')
            ...
            feed_dict = {x:(indices, values, shape)}
            ...
    '''
    vocab_dict = vocab_processor.vocabulary_._mapping # vocab_dict 必须fit的是unicode编码
    sparse_values = []
    all_nums = []
    sparse_indices = []
    left_start = 0
    for text in x_text:
        text_IDs = [] #当前问句的每个word的subwords在词表中的id, list, 对应sparse_values
        cur_nums = [] #当前问句的每个word的subword个数
        cur_indices = [] #当前每个word(subword)的稀疏索引, 对应sparse_indices

        #word's sparse_indices and sparse_values
        words = text.split(' ')
        for word_bias, word_uni in enumerate(words):
            #get word_subwords_vocab_indices
            subwords, subword_num = generate_subword(word_uni, max_num)
            subwords_str = ' '.join(subwords)
            word_subword_indices = [vocab_dict[word_uni] if word_uni in vocab_dict else 0]
            subword_indices = [vocab_dict[i.decode('utf-8')] if i.decode('utf-8') in vocab_dict else 0 for i in subwords]
            word_subword_indices.extend(subword_indices)
            
            #word_subword sparse indices
            for right_start in xrange(len(word_subword_indices)):
                cur_indices.append([left_start + word_bias, right_start])
            cur_nums.append(subword_num + 1)
            text_IDs.extend(word_subword_indices)
        
        #padding words' sparse_indices and sparse_values
        for padding_word_bias in xrange(len(words), seq_len):
            text_IDs.append(0)
            cur_bias = left_start + padding_word_bias
            cur_indices.append([cur_bias, 0])
            cur_nums.append(1)
        
        left_start += seq_len
        sparse_indices.extend(cur_indices)
        sparse_values.extend(text_IDs)
        all_nums.extend(cur_nums)
    return sparse_indices, sparse_values, all_nums





def recomb_sent_with_subword(words, max_num=8):
    new_words = []
    for word_uni in words:
        word_len = len(word_uni)
        new_words.append(word_uni)
        if word_uni in special_words:
            break
        grams, _ = generate_subword(word_uni, max_num)
        grams_uni = [i.decode('utf-8') for i in grams]
        new_words.extend(grams_uni)
    return new_words




def expand_batch_sents_with_subword(texts, batch_size, max_num=8):
    new_texts = []
    for text_str in texts:
        text = text_str.split(' ')
        new_text = recomb_sent_with_subword(text, max_num)
        num_iter = int(len(new_text) - 1) / batch_size + 1
    
        for batch_num in range(num_iter):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, len(new_text))
            cur_batch = new_text[start_index:end_index]
            cur_batch_str = ' '.join(cur_batch)
            new_texts.append(cur_batch_str)
    return new_texts




def get_stopwords_set(stop_words_file):
    '''
    @breif: read stop_words file
    '''
    stop_set = set()
    with codecs.open(stop_words_file, 'r', 'utf8') as fr:
        for word in fr:
            word = word.strip()
            stop_set.add(word)
    return stop_set


def one_hot_encode(list):
    array = np.array(list)
    max_class = array.max() + 1
    return np.eye(max_class)[array]

def load_bin_vec(file_name, vocab, ksize=100):
    time_str = datetime.datetime.now().isoformat()
    print('{}:开始筛选数据词汇...'.format(time_str))

    word_vecs = {}
    #model = gensim.models.Word2Vec.load_word2vec_format(file_name, binary=True)
    model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=True)
    #model = gensim.models.KeyedVectors.load_word2vec_format(file_name, binary=False)
    for word in vocab:
        try:
            word_vecs[word] = model[word]
        except:
            word_vecs[word] = np.random.uniform(-1.0, 1.0, ksize).astype(np.float32)
    return word_vecs



def get_word_vecs(word_vecs_path, vocab, vocab_idx_map, k, is_random=False):
    word_vecs = load_bin_vec(word_vecs_path, vocab, k)
    time_str = datetime.datetime.now().isoformat()
    print('{}:生成嵌入层参数W...'.format(time_str))
    
    vocab_size = len(word_vecs)
    W = np.random.uniform(-1.0, 1.0, size=[vocab_size, k]).astype(np.float32)
    if not is_random:
        print('非随机初始化')
        for i, word in enumerate(word_vecs):
            idx = vocab_idx_map[word]
            W[idx] = word_vecs[word]
    time_str = datetime.datetime.now().isoformat()
    print("{}:生成嵌入层参数W完毕".format(time_str))
    return W


def write_label_file(label2idx, output_file):
    dir_name = os.path.dirname(output_file)
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
    with open(output_file, 'w') as fw:
        labels = sorted(label2idx.items(), key=lambda x:x[1])
        for label, value in labels:
            fw.write(label + '\t' + str(value) + '\n')





def read_labels_file(label_file):
    #get real label
    idx2label = {}
    label2idx = {}
    with codecs.open(label_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            line_info = line.split('\t')
            label = line_info[0]
            label_idx = line_info[1]
            idx2label[int(label_idx)] = label 
            label2idx[label] = int(label_idx)
    return label2idx, idx2label

def read_code_file(code_file):
    #get real label
    label2code = {}
    with open(code_file, 'r') as fr:
        for line in fr:
            line = line.strip().decode('utf8')
            line_info = line.split('\t')
            label = line_info[0]
            code_value = line_info[1]
            label2code[label] = code_value
    return label2code



def process_sentence(text, stop_set, label2idx):
    uni_sents = []
    sent_segs = []
    sent, word_segs_str = stringhandler.split_word_and_seg(text, stop_set)
    uni_sents.append(sent.decode('utf-8'))
    sent_segs.append(word_segs_str)
    return uni_sents, sent_segs



def load_test_data(data_file):
    uni_sents = []
    sent_segs = []
    labels = []
    stop_set = get_stop_words_set(STOP_WORDS_FILE)
    with open(data_file, 'r') as lines:
        for line in lines:
            line = line.strip().lower()
            line_info = line.split('\t')
            trunks_str = line_info[0]
            sent, word_segs_str = stringhandler.split_word_and_seg(trunks_str, stop_set)
            uni_sents.append(sent.decode('utf-8'))
            sent_segs.append(word_segs_str)

            label = line_info[1]
            labels.append(label)

    label2idx, _ = read_labels_file(LABEL_FILE)
    label_indices = [label2idx[label] if label in label2idx else 0 for label in labels]
    one_hot_labels = one_hot_encode(label_indices)
    return [uni_sents, sent_segs, one_hot_labels]

def load_data_and_labels(data_file):
    stop_set = get_stop_words_set(STOP_WORDS_FILE)
    uni_sents = []
    sent_segs = []
    labels = []
    enum_index = 0
    label2idx = {}
    with open(data_file, 'r') as lines:
        for line in lines:
            line = line.strip().lower()
            line_info = line.split('\t')
            if len(line_info) < 2:
                continue
            trunks_str = line_info[0]
            sent, word_segs_str = stringhandler.split_word_and_seg(trunks_str, stop_set)
            uni_sents.append(sent.decode('utf-8'))
            sent_segs.append(word_segs_str)
            label = line_info[1]
            labels.append(label)
            if label not in label2idx:
                label2idx[label] = enum_index
                enum_index += 1

    label_indices = [label2idx[label] for label in labels]
    one_hot_labels = one_hot_encode(label_indices)
    #print('one hot labels:', one_hot_labels)
    write_label_file(label2idx, LABEL_FILE)
    return [uni_sents, sent_segs, one_hot_labels]


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indice = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indice]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def batch_iter2(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    data = one_hot_encode(data)
    rs = []
    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indice = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indice]
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            rs.append(shuffled_data[start_index:end_index])
    return rs

if __name__ == '__main__':
    print(embed)
    #batch_embed = tf.split(embed, 2)
    batch_embed = tf.reshape(embed, [-1, 5, 10])
    #batch_embed = tf.expand_dims(batch_embed, -1)
    #batch_embed = tf.concat(batch_embed, -1)
    rs = batch_iter2(batch_embed)
    print(rs)
	
