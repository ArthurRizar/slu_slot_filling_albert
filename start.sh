#coding:utf-8
###################################################
# File Name: start.sh
# Author: Meng Zhao
# mail: @
# Created Time: 2019年10月31日 星期四 16时48分31秒
#=============================================================
source activate tensorflow_new_3.6
export MODEL_DIR=/root/zhaomeng/albert_test/pretrain_models/albert_tiny_489k

#export MODEL_DIR=/root/zhaomeng/albert_test/pretrain_models/albert_base_zh_additional_36k_steps
#export MODEL_DIR=/root/zhaomeng/albert_test/pretrain_models/albert_small_zh_google

#export MODEL_DIR=/home/zhaomeng/albert_model/albert_tiny_489k

python -m models.albert.run_sequencelabeling --task_name=ner \
                         --output_dir=./output \
                         --data_dir=./data \
                         --init_checkpoint=$MODEL_DIR/albert_model.ckpt \
                         --bert_config_file=$MODEL_DIR/albert_config.json \
                         --vocab_file=$MODEL_DIR/vocab.txt \
                         --max_seq_length=64  \
                         --num_train_epochs=20 \
                         --learning_rate=5e-4  \
                         --train_batch_size=64 \
                         --do_predict=true \
                         --do_eval=true \
                         --do_train=true \

#2e-5, 6e-5, 1e-4, 5e-6, 5e-5

#python run_classifier.py --task_name=sentence_pair \
#                         --output_dir=./output \
#                         --data_dir=./data \
#                         --init_checkpoint=$MODEL_DIR/albert_model.ckpt \
#                         --bert_config_file=$MODEL_DIR/albert_config.json \
#                         --vocab_file=$MODEL_DIR/vocab.txt \
#                         --max_seq_length=128  \
#                         --num_train_epochs=5 \
#                         --learning_rate=1e-4  \
#                         --train_batch_size=64 \
#                         --do_train=true \
#                         --do_eval=true \
#                         --do_predict=true

