###################################################################
# File Name: tf_serving_stop.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年12月09日 星期一 19时33分07秒
#=============================================================
#!/bin/bash
CLIENT_NAME=tf_serving_ner_http.py
ps -ef | grep "python $CLIENT_NAME" | grep -v 'grep' 
ps -ef | grep "python $CLIENT_NAME" | grep -v 'grep' | awk '{print $2}' | xargs -n1 kill -9


SERVER_PORT=16378
ps -ef | grep "tensorflow_model_server --rest_api_port=$SERVER_PORT" | grep -v 'grep' 
ps -ef | grep "tensorflow_model_server --rest_api_port=$SERVER_PORT" | grep -v 'grep' | awk '{print $2}' | xargs -n1 kill -9
