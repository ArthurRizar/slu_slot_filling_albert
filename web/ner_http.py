#coding:utf-8
###################################################
# File Name: httpserver.py
# Author: Meng Zhao
# mail: @
# Created Time: 2018年06月06日 星期三 15时24分44秒
#=============================================================

import os
import sys
import ujson as json
import tornado.web
import tornado.ioloop


import numpy as np
import datetime
import logging
import traceback


sys.path.append('../')


import evaluator

#from tornado.concurrent import run_on_executor
#from concurrent.futures import ThreadPoolExecutor

from pathlib import Path
from multiprocessing import Pool, TimeoutError

from setting import *
from evaluator import Evaluator
from preprocess import dataloader
from common.segment.segment_client import SegClient


os.environ["CUDA_VISIBLE_DEVICES"] = "" #不使用GPU




class NERHttpServer(tornado.web.RequestHandler):

    def initialize(self, pred_instance):
        self.pred_instance = pred_instance

    @tornado.gen.coroutine
    def head(self):
        self.write('OK')

            
    def prepare(self):
        start_time = datetime.datetime.now()
        if self.request.headers.get("Content-Type", "").startswith("application/json"):
            self.data = json.loads(self.request.body)
        else:
            self.data = {}
            for k in self.request.arguments:
                self.data[k] = self.get_argument(k)
        end_time = datetime.datetime.now()
        cost = (end_time - start_time).total_seconds() * 1000
        logging.info('json convert cost' + str(cost))

    @tornado.gen.coroutine   
    def main_process(self):
        err_dict = {}
        try:
            start_time = datetime.datetime.now()
            text = self.data['text']
            response = self.pred_instance.evaluate(text)

            end_time = datetime.datetime.now()
            cost = (end_time - start_time).total_seconds() * 1000
            logging.info('evaluating cost:' + str(cost))

            response_json = json.dumps(response, ensure_ascii=False)
            self.write(response_json)

        except Exception as err:
            err_dict['errMsg'] = traceback.format_exc()
            self.write(json.dumps(err_dict, ensure_ascii=False))
            logging.warning(traceback.format_exc())

    def get(self):
        self.main_process()

    def post(self):
        self.main_process()




if __name__ == '__main__':


    config = {}
    #MODEL_DIR = '../output.1205_1e-4_base' 
    #CHECKPOINT_DIR = '../output.1205_1e-4_base/checkpoints'

    config['model_dir'] = MODEL_DIR
    config['max_seq_length'] = 64
    config['label_map_file'] = MODEL_DIR + '/label_map'
    config['vocab_file'] = MODEL_DIR + '/vocab.txt'

    subdirs = [x for x in Path(CHECKPOINT_DIR).iterdir()
                            if x.is_dir() and 'temp' not in str(x)]
    latest = str(sorted(subdirs)[-1])
    config['model_pb_path'] = latest


    pred_instance = Evaluator(config)
    pred_instance.evaluate('我明天去万达')


    application = tornado.web.Application([
            (r"/ner", NERHttpServer,
            dict(pred_instance=pred_instance))
        ])
    application.listen(TF_SERVING_CLIENT_PORT)
   
    tornado.ioloop.IOLoop.instance().start()
