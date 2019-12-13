#coding:utf-8
###################################################
# File Name: start.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月13日 星期三 11时30分49秒
#=============================================================
source activate tensorflow_new_3.6

export REST_API_PORT=16378
export MODEL_DIR=$PWD/../output.1205_1e-4_base/checkpoints/
export MODEL_NAME=default


nohup tensorflow_model_server --rest_api_port=$REST_API_PORT \
                              --model_name=$MODEL_NAME \
                              --model_base_path=$MODEL_DIR \
      >output.file 2>&1 &
sleep 5
nohup python tf_serving_ner_http.py> output.file 2>&1 &
