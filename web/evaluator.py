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
import logging
import codecs
import json
import numpy as np


sys.path.append('../')

from pathlib import Path
from preprocess import bert_data_utils
from preprocess import dataloader
from preprocess import tokenization
from preprocess import ner_utils
from tensorflow.contrib import learn
from tensorflow.contrib import predictor
from setting import *


#os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU



class Evaluator(object):
    def __init__(self, config):
        self.max_seq_length = config['max_seq_length']
        self.vocab_file = config['vocab_file']
        self.label_map_file = config['label_map_file']

        self.model_dir = config['model_dir']
        self.max_seq_length = config['max_seq_length']
        self.vocab_file = config['vocab_file']
        self.model_pb = config['model_pb_path']


        label2idx, idx2label = bert_data_utils.read_ner_label_map_file(self.label_map_file)
        self.idx2label = idx2label
        self.label2idx = label2idx

        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)

        self.stop_set = dataloader.get_stopwords_set(STOPWORD_FILE)

        self.predict_fn = predictor.from_saved_model(self.model_pb)



    def evaluate(self, text):
        input_ids, input_mask, segment_ids = self.trans_text2ids(text)

        feed_dict = {'input_ids': input_ids,
                     'input_mask': input_mask,
                     'segment_ids': segment_ids}
        start_time = datetime.datetime.now()
        predict_result = self.predict_fn(feed_dict)
        end_time = datetime.datetime.now()
        print('cost:', end_time - start_time)
        pred_label_ids = predict_result['pred_label_ids']

        tags = [self.idx2label[t].upper() for t in pred_label_ids[0]]
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
        return input_ids, input_mask, segment_ids 


if __name__ == '__main__':

    #MODEL_DIR = '../runs/saved_model'
    #VOCAB_FILE = '../runs/vocab.txt'
    #LABEL_MAP_FILE = '../runs/label_map'
    config = {}
    config['model_dir'] = MODEL_DIR

    subdirs = [x for x in Path(CHECKPOINT_DIR).iterdir()
                    if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    config['model_pb_path'] = latest
    config['max_seq_length'] = 64
    config['vocab_file'] = VOCAB_FILE
    config['label_map_file'] = LABEL_MAP_FILE


    pred_instance = Evaluator(config)
    for i in range(3):
        rs = pred_instance.evaluate('班车报表')
    rs = pred_instance.evaluate('明天我要上班')

    json_str = json.dumps(rs, indent=2, ensure_ascii=False)
    print(json_str)

    
    rs = pred_instance.evaluate('马思文要跟老板去上海的田亩公司')
    print(rs)

    rs = pred_instance.evaluate('周冰冰要跟老板去黄埔区的田亩公司')
    pred_instance.evaluate('浙江广播电视大学项目验证会')

