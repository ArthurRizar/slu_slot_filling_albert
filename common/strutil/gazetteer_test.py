#coding:utf-8
###################################################
# File Name: gazetteer_test.py
# Author: Meng Zhao
# mail: @
# Created Time: 2020年05月08日 星期五 14时20分11秒
#=============================================================
import codecs    
import collections
import gensim
import numpy as np
import tensorflow as tf
import gazetteer
from trie import Trie
from alphabet import Alphabet

tf.enable_eager_execution()





def test_gazetteer(gaz_file, input_file, number_normalized=False):
    gaz_count = collections.defaultdict(int)
    gaz = gazetteer.Gazetteer()
    gaz.build_gaz_from_file(gaz_file)
    alphabet = Alphabet('gaz')
    with codecs.open(input_file, 'r', 'utf8') as fr:
        word_list = []
        for line in fr:
            line = line.strip()
            if len(line) > 3:
                word = line.split()[0]
                if number_normalized:
                    word = gazetteer.normalize_word(word)
                word_list.append(word)
            else:
                entities = gazetteer.match_all_entities(word_list, gaz, gaz_count, alphabet)
                gazetteer.remove_covered_entities(entities, gaz_count)
                word_list = []

    print(gaz_count)
    print(gaz)
    print(alphabet)
    return gaz, alphabet, gaz_count


def test_generate_simple_example(input_file, gaz, gaz_alphabet, gaz_count, word_alphabet):
    instances = []
    with codecs.open(input_file, 'r', 'utf8') as fr:
        words = []
        labels = []
        for line in fr:
            line = line.strip()
            line_info = line.split()
            if len(line_info) == 2:
                word = line_info[0]
                label = line_info[1]
                words.append(word)
                labels.append(label)
            else:
                input_gaz_layer, input_gaz_layer_mask, input_gaz_count = gazetteer.generate_gaz_example(words, gaz, gaz_count, gaz_alphabet)
                print(input_gaz_layer)
                print(input_gaz_layer_mask)
                print(np.shape(input_gaz_layer), np.shape(input_gaz_layer_mask))
                exit()
                words = []
                labels = []
                instances.append([input_gaz_layer, input_gaz_count, input_gaz_layer_mask])



def test_generate_example(input_file, gaz, gaz_alphabet, gaz_count, word_alphabet):
    instances = []
    with codecs.open(input_file, 'r', 'utf8') as fr:
        words = []
        labels = []
        for line in fr:
            line = line.strip()
            line_info = line.split()
            if len(line_info) == 2:
                word = line_info[0]
                label = line_info[1]
                words.append(word)
                labels.append(label)
            else:
                word_len = len(words)
                input_gaz_ids = []
                input_gaz_layer_mask = []
                input_gaz_char_mask = []
                
                input_gaz_layer = [[[] for __ in range(4)] for _ in range(word_len)]
                input_gaz_count = [[[] for __ in range(4)] for _ in range(word_len)]
                input_gaz_char_ids = [[[] for __ in range(4)] for _ in range(word_len)]

                max_gaz_list = 0
                max_gaz_char_len = 0
                for idx in range(word_len):
                    matched_list = gaz.enumerate_match_list(words[idx:])
                    matched_length = [len(entity) for entity in matched_list]
                    matched_ids = [gaz_alphabet.get_index(entity) for entity in matched_list]

                    if matched_length:
                        max_gaz_char_len = max(max(matched_length), max_gaz_char_len)

                    for w_len, matched_idx, entity in zip(matched_length, matched_ids, matched_list):
                        cur_gaz_char_ids = []
                        for char in entity:
                            cur_gaz_char_ids.append(word_alphabet.get_index(char))
                        if w_len == 1:
                            input_gaz_layer[idx][3].append(matched_idx)
                            input_gaz_count[idx][3].append(1)
                            input_gaz_char_ids[idx][3].append(cur_gaz_char_ids)
                        else:
                            input_gaz_layer[idx][0].append(matched_idx)
                            input_gaz_count[idx][0].append(gaz_count[matched_idx])
                            input_gaz_char_ids[idx][0].append(cur_gaz_char_ids)
                            input_gaz_layer[idx + w_len - 1][2].append(matched_idx)
                            input_gaz_count[idx + w_len - 1][2].append(gaz_count[matched_idx])
                            input_gaz_char_ids[idx + w_len - 1][2].append(cur_gaz_char_ids)
                            for cur_len in range(w_len - 2):
                                input_gaz_layer[idx + cur_len + 1][1].append(matched_idx)
                                input_gaz_count[idx + cur_len + 1][1].append(gaz_count[matched_idx])
                                input_gaz_char_ids[idx + cur_len + 1][1].append(cur_gaz_char_ids)
                        

                    for label in range(4):
                        if not input_gaz_layer[idx][label]:
                            input_gaz_layer[idx][label].append(0)
                            input_gaz_count[idx][label].append(1)
                            input_gaz_char_ids[idx][label].append([0])
                        max_gaz_list = max(len(input_gaz_layer[idx][label]), max_gaz_list)

                    matched_ids = [gaz_alphabet.get_index(entity) for entity in matched_list]
                    if matched_ids:
                        input_gaz_ids.append([matched_ids, matched_length])
                    else:
                        input_gaz_ids.append([])

                for idx in range(word_len):
                    gaz_mask = []
                    gaz_char_mask = []

                    for label in range(4):
                        label_len = len(input_gaz_layer[idx][label])
                        count_set = set(input_gaz_count[idx][label])
                        if len(count_set) == 1 and 0 in count_set:
                            input_gaz_count[idx][label] = [1] * label_len

                        mask = [0] * label_len + [1] * (max_gaz_list - label_len)
                        input_gaz_layer[idx][label] += [0] * (max_gaz_list - label_len) # padding
                        input_gaz_count[idx][label] += [0] * (max_gaz_list - label_len) # padding
                        
                        char_mask = []
                        for g in range(len(input_gaz_char_ids[idx][label])):
                            g_len = len(input_gaz_char_ids[idx][label][g])
                            cur_char_mask = [0] * g_len + [1] * (max_gaz_char_len - g_len)
                            char_mask.append(cur_char_mask)
                            input_gaz_char_ids[idx][label][g] += [0] * (max_gaz_char_len - g_len)
                        input_gaz_char_ids[idx][label] += [[0 for _ in range(max_gaz_char_len)]] * (max_gaz_list - label_len)
                        
                        char_mask += [[1 for _ in range(max_gaz_char_len)]] * (max_gaz_list - label_len)
                        
                        gaz_mask.append(mask)
                        gaz_char_mask.append(char_mask)
                    input_gaz_layer_mask.append(gaz_mask)
                    input_gaz_char_mask.append(gaz_char_mask)

                instances.append([input_gaz_ids, input_gaz_layer, input_gaz_count, input_gaz_char_ids, input_gaz_layer_mask, input_gaz_char_mask])

                print(input_gaz_layer)
                print(input_gaz_layer_mask)
                print(np.shape(input_gaz_layer), np.shape(input_gaz_layer_mask))
                exit()

                words = []
                labels = []
    print(np.shape(instances))     
    #print(instances)
    return instances 

                        

def test_generate_example1(input_file, gaz, gaz_alphabet,
					 gaz_count, word_alphabet, max_sent_length=128, char_padding_size=-1, char_padding_symbol = '</pad>'):

    in_lines = open(input_file,'r',encoding="utf-8").readlines()
    instance_texts = []
    instance_Ids = []
    words = []
    labels = []
    for idx in range(len(in_lines)):
        line = in_lines[idx]
        if len(line) > 2:
            pairs = line.strip().split()
            word = pairs[0]
            label = pairs[-1]
            words.append(word)
            labels.append(label)
        else:
            if ((max_sent_length < 0) or (len(words) < max_sent_length)) and (len(words)>0):
                gaz_Ids = []
                layergazmasks = []
                gazchar_masks = []
                w_length = len(words)

                gaz_layer = [ [[] for i in range(4)] for _ in range(w_length)]  # gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[w_id1,w_id2,...]  None:0
                gazs_count = [ [[] for i in range(4)] for _ in range(w_length)]

                gaz_char_Id = [ [[] for i in range(4)] for _ in range(w_length)]  ## gazs:[c1,c2,...,cn]  ci:[B,M,E,S]  B/M/E/S :[[w1c1,w1c2,...],[],...]

                max_gazlist = 0
                max_gazcharlen = 0
                for idx in range(w_length):

                    matched_list = gaz.enumerate_match_list(words[idx:])
                    matched_length = [len(a) for a in matched_list]
                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]

                    if matched_length:
                        max_gazcharlen = max(max(matched_length),max_gazcharlen)


                    for w in range(len(matched_Id)):
                        gaz_chars = []
                        g = matched_list[w]
                        for c in g:
                            gaz_chars.append(word_alphabet.get_index(c))

                        if matched_length[w] == 1:  ## Single
                            gaz_layer[idx][3].append(matched_Id[w])
                            gazs_count[idx][3].append(1)
                            gaz_char_Id[idx][3].append(gaz_chars)
                        else:
                            gaz_layer[idx][0].append(matched_Id[w])   ## Begin
                            gazs_count[idx][0].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx][0].append(gaz_chars)
                            wlen = matched_length[w]
                            gaz_layer[idx+wlen-1][2].append(matched_Id[w])  ## End
                            gazs_count[idx+wlen-1][2].append(gaz_count[matched_Id[w]])
                            gaz_char_Id[idx+wlen-1][2].append(gaz_chars)
                            for l in range(wlen-2):
                                gaz_layer[idx+l+1][1].append(matched_Id[w])  ## Middle
                                gazs_count[idx+l+1][1].append(gaz_count[matched_Id[w]])
                                gaz_char_Id[idx+l+1][1].append(gaz_chars)

                    for label in range(4):
                        if not gaz_layer[idx][label]:
                            gaz_layer[idx][label].append(0)
                            gazs_count[idx][label].append(1)
                            gaz_char_Id[idx][label].append([0])

                        max_gazlist = max(len(gaz_layer[idx][label]),max_gazlist)

                    matched_Id  = [gaz_alphabet.get_index(entity) for entity in matched_list]  #词号
                    if matched_Id:
                        gaz_Ids.append([matched_Id, matched_length])
                    else:
                        gaz_Ids.append([])

                ## batch_size = 1
                for idx in range(w_length):
                    gazmask = []
                    gazcharmask = []

                    for label in range(4):
                        label_len = len(gaz_layer[idx][label])
                        count_set = set(gazs_count[idx][label])
                        if len(count_set) == 1 and 0 in count_set:
                            gazs_count[idx][label] = [1]*label_len

                        mask = label_len*[0]
                        mask += (max_gazlist-label_len)*[1]

                        gaz_layer[idx][label] += (max_gazlist-label_len)*[0]  ## padding
                        gazs_count[idx][label] += (max_gazlist-label_len)*[0]  ## padding

                        char_mask = []
                        for g in range(len(gaz_char_Id[idx][label])):
                            glen = len(gaz_char_Id[idx][label][g])
                            charmask = glen*[0]
                            charmask += (max_gazcharlen-glen) * [1]
                            char_mask.append(charmask)
                            gaz_char_Id[idx][label][g] += (max_gazcharlen-glen) * [0]
                        gaz_char_Id[idx][label] += (max_gazlist-label_len)*[[0 for i in range(max_gazcharlen)]]
                        char_mask += (max_gazlist-label_len)*[[1 for i in range(max_gazcharlen)]]

                        gazmask.append(mask)
                        gazcharmask.append(char_mask)
                    layergazmasks.append(gazmask)
                    gazchar_masks.append(gazcharmask)

                print(gaz_layer)
                print(layergazmasks)
                print(np.shape(gaz_layer), np.shape(layergazmasks))
                print(layergazmasks)
                exit()

                instance_texts.append([words, gaz_layer, labels])
                instance_Ids.append([gaz_Ids, gaz_layer, gazs_count, gaz_char_Id, layergazmasks,gazchar_masks])

            words = []
            labels = []
    print(np.shape(instance_Ids))
    #print(instance_Ids)
    return instance_Ids


def test_tf_process(instances, embedding_table, use_count=True):
    gaz_embedding_table = tf.Variable(embedding_table, name='W', trainable=True)
    for item in instances:
        gazs = [item[0]]
        layer_gazs = [item[1]]
        gaz_count = [item[2]]
        gaz_char_ids = [item[3]]
        input_gaz_mask = [item[4]]
        gaz_char_mask = [item[5]]

        print(gazs)
        print(gaz_count)
        print(gaz_char_ids)
        print(input_gaz_mask)
        print(gaz_char_mask)

        gaz_num = len(layer_gazs[0][0])
        max_gaz_num = gaz_num
        print('gaz:num:', gaz_num)
        print('max_gaz_num:', max_gaz_num)

        gaz_len = len(gaz_char_ids[0][0][0])
        max_gaz_len = gaz_len
        print('gaz_len:', gaz_len)
        print('max_gaz_len:', max_gaz_len)
        
        gaz_chars_tensor = tf.zeros(shape=(1, 128, 4, max_gaz_num, max_gaz_len))
        gaz_mask_tensor = tf.ones(shape=(1, 128, 4, max_gaz_num))
        gazchar_mask_tensor = tf.ones(shape=(1, 128, 4, max_gaz_num, max_gaz_len))
        print(gaz_chars_tensor)
        print(gaz_mask_tensor)

        print(layer_gazs)
        gaz_embeddings = tf.nn.embedding_lookup(gaz_embedding_table, layer_gazs)
        print(gaz_embeddings)

        gaz_mask = tf.expand_dims(input_gaz_mask, axis=-1)
        print(gaz_mask)

        gaz_mask = tf.tile(gaz_mask, [1, 1, 1, 1, 50])
        print(gaz_mask)
        gaz_mask = 1 - gaz_mask
        print(gaz_mask)

        gaz_mask = tf.cast(gaz_mask, dtype=tf.float64)

        # (batch, term, num_gaz_class, layer, gaz_hidden_dim)
        gaz_embeddings = gaz_embeddings * gaz_mask
        print(gaz_embeddings)

        use_count = False
        if use_count:
            count_num = tf.reduce_sum(gaz_count, axis=3, keep_dims=True)
            print(np.shape(count_num))
            count_num = tf.reduce_sum(count_num, axis=2, keep_dims=True)
            print(np.shape(count_num))

            weights = tf.truediv(gaz_count, count_num)
            print(np.shape(weights))
            weights *= 4

            print(weights)
            weights = tf.expand_dims(weights, axis=-1)
            print(np.shape(weights))

            gaz_embeddings = gaz_embeddings * weights
            print(np.shape(gaz_embeddings))
        else:
            gaz_num = tf.cast(tf.reduce_sum(1 - np.array(input_gaz_mask), axis=-1, keep_dims=True), dtype=tf.float64)
            print(gaz_num)
            print(np.shape(gaz_num))
            gaz_embeddings = tf.reduce_sum(gaz_embeddings, axis=-2) / gaz_num
            print(np.shape(gaz_embeddings))
            print(gaz_embeddings)
        
        _, max_seq_length, num_gaz_class, gaz_hidden_dim = np.shape(gaz_embeddings)
        gaz_embeddings_cat = tf.reshape(gaz_embeddings, [-1, max_seq_length, num_gaz_class * gaz_hidden_dim])
        print(gaz_embeddings_cat)
        exit()



if __name__ == '__main__':
    embeddings_file = '../../embeddings/ctb.50d.vec'
    embeddings_file = '../../embeddings/gigaword_chn.all.a2b.uni.ite50.vec'
    #input_file = '../../data/train.tsv'
    input_file = 'test.tsv'
    gaz, alphabet, gaz_count = test_gazetteer(embeddings_file, input_file)
    #gaz, alphabet, gaz_count = gazetteer.build_gazetteer_and_counter_from_file(embeddings_file, input_file)
    embedding_table = gazetteer.build_gaz_pretrain_emb(embeddings_file, alphabet)
    #instances = test_generate_example(input_file, gaz, alphabet, gaz_count, alphabet)
    instances = test_generate_simple_example(input_file, gaz, alphabet, gaz_count, alphabet)
    exit()
    test_tf_process(instances, embedding_table)
