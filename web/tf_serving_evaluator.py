#coding:utf-8
###################################################
# File Name: eval.py
# Author: Meng Zhao
# mail: @
# Created Time: Fri 23 Mar 2018 09:27:09 AM CST
#=============================================================
import os
import sys
import csv

import datetime
import json
import logging
import numpy as np
import requests
import codecs


sys.path.append('../')

from preprocess import bert_data_utils
from preprocess import dataloader
from preprocess import tokenization
from preprocess import ner_utils
from setting import *
from tensorflow.contrib import learn


#os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU


def request_model_predict(url, input_ids, input_mask, segment_ids, is_training, signature_name):
    data = {'inputs': {'input_ids': input_ids,
                       'input_mask': input_mask,
                       'segment_ids': segment_ids}}


    json_data = json.dumps(data)

    #print(json_data)
    #results = requests.post(url, data=json_data)
    results = requests.post(url, data=json_data)
    #print(results.text)
    result_json = json.loads(results.text)
    #print('outputs' not in result_json)
    #print(result_json)
    if 'outputs' not in result_json:
        raise Exception(str(result_json))
    pred_labels = result_json['outputs']['pred_label_ids']
    probs = result_json['outputs']['probabilities']
    return np.array(pred_labels)


class Evaluator(object):
    def __init__(self, config):
        #self.top_k_num = config['top_k']
        self.model_dir = config['model_dir']
        self.max_seq_length = config['max_seq_length']
        self.vocab_file = config['vocab_file']
        self.label_map_file = config['label_map_file']
        
        self.url = config['tf_serving_url']
        self.signature_name = config['signature_name']


        #init label dict and processors
        label2idx, idx2label = bert_data_utils.read_ner_label_map_file(self.label_map_file)
        self.idx2label = idx2label
        self.label2idx = label2idx
        

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

    
        #init stop set
        self.stop_set = dataloader.get_stopwords_set(STOPWORD_FILE)

   

    def close_session(self):
        self.sess.close()

    


    def evaluate(self, text):
        input_ids, input_mask, segment_ids = self.trans_text2ids(text)

        cur_pred_labels = request_model_predict(self.url, input_ids, input_mask, segment_ids, False, self.signature_name)

        print(cur_pred_labels)
        tags = [self.idx2label[t].upper() for t in cur_pred_labels[0]]
        print(tags, len(tags))
        tags = tags[1: len(text) + 1]
        print(text, len(text))
        print(tags, len(tags))
        tags = ner_utils.bert_result_to_json(text, tags) 
        print(text, len(text))
        print(tags, len(tags))

        return tags 


    def trans_text2ids(self, text):
        if text[-1] in self.stop_set:
            text = text[: -1]
        example = bert_data_utils.InputExample(guid='1', text_a=text)
        seq_length = min(self.max_seq_length, len(text) + 2)
        #seq_length = self.max_seq_length
        feature = bert_data_utils.convert_online_example(example,
                                                seq_length, self.tokenizer)
        input_ids = [feature.input_ids]
        input_mask = [feature.input_mask]
        segment_ids = [feature.segment_ids]
        #print(input_ids)
        #print(input_mask)
        #print(segment_ids)
        return input_ids, input_mask, segment_ids 


if __name__ == '__main__':
    config = {}

    config['model_dir'] = MODEL_DIR 
    config['max_seq_length'] = 64
    config['label_map_file'] = MODEL_DIR + '/label_map'
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'

    config['tf_serving_url'] = 'http://localhost:'+str(TF_SERVING_REST_PORT)+'/v1/models/default:predict'
    config['signature_name'] = 'predict_text'



    pred_instance = Evaluator(config)
    rs = pred_instance.evaluate('明天我要去万达')
    print(rs)
    print(np.shape(rs))
    
    rs = pred_instance.evaluate('马思文要跟老板去上海的田亩公司')
    print(rs)

