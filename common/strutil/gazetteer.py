#coding:utf-8
###################################################
# File Name: gazetteer.py
# Author: Meng Zhao
# mail: @
# Created Time: 2020年05月08日 星期五 14时20分11秒
#=============================================================
import codecs    
import collections
import gensim
import numpy as np
import tensorflow as tf
from .trie import Trie
from .alphabet import Alphabet

tf.enable_eager_execution()

class Gazetteer:
    def __init__(self, lower=False):
        self.trie = Trie()
        self.ent2type = {} ## word list to type
        self.ent2id = {"<UNK>": 0}   ## word list to id
        self.lower = lower
        self.space = ""

    def enumerate_match_list(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        match_list = self.trie.enumerate_match(word_list, self.space)
        return match_list

    def insert(self, word_list, source):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        self.trie.insert(word_list)
        string = self.space.join(word_list)
        if string not in self.ent2type:
            self.ent2type[string] = source
        if string not in self.ent2id:
            self.ent2id[string] = len(self.ent2id)

    def search_id(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2id:
            return self.ent2id[string]
        return self.ent2id["<UNK>"]

    def search_type(self, word_list):
        if self.lower:
            word_list = [word.lower() for word in word_list]
        string = self.space.join(word_list)
        if string in self.ent2type:
            return self.ent2type[string]
        print("Error in finding entity type at gazetteer.py, exit program! String:", string)
        exit(0)

    def size(self):
        return len(self.ent2type)


    def build_gaz_from_file(self, gaz_file):
        if gaz_file is None:
            print("Gaz file is None, load nothing")
            return

        with codecs.open(gaz_file, 'r', 'utf8') as fr:
            for line in fr:
                line = line.strip()
                if line == "":
                    continue
                infos = line.split()
                if len(infos) == 2:
                    continue
                word = infos[0]
                self.insert(word, 'one_source')
            print('Load gaz file: ', gaz_file, ", total size: ", self.size())

def build_gazetteer_and_counter_from_file(gaz_file, input_file, number_normalized=False):
    gaz_count = collections.defaultdict(int)
    gaz = Gazetteer()
    gaz.build_gaz_from_file(gaz_file)
    alphabet = Alphabet('gaz')
    with codecs.open(input_file, 'r', 'utf8') as fr:
        word_list = []
        for line in fr:
            line = line.strip()
            if len(line) > 3:
                word = line.split()[0]
                if number_normalized:
                    word = normalize_word(word)
                word_list.append(word)
            else:
                entities = match_all_entities(word_list, gaz, gaz_count, alphabet)
                remove_covered_entities(entities, gaz_count)
                word_list = []
    print('alphabet size:', alphabet.size())
    print('-' * 20)
    return gaz, alphabet, gaz_count


def pad_gaz_example(seq_len, input_gaz_layer, input_gaz_layer_mask, input_gaz_count, max_gaz_list):
    for idx in range(seq_len):
        gaz_mask = []
        for flag_idx in range(4):
            label_len = min(len(input_gaz_layer[idx][flag_idx]), max_gaz_list)
            count_set = set(input_gaz_count[idx][flag_idx])
            if len(count_set) == 1 and 0 in count_set:
                input_gaz_count[idx][flag_idx] = [1] * label_len
            input_gaz_layer[idx][flag_idx] = input_gaz_layer[idx][flag_idx][: max_gaz_list]
            input_gaz_count[idx][flag_idx] = input_gaz_count[idx][flag_idx][: max_gaz_list]
            input_gaz_layer[idx][flag_idx] += [0] * (max_gaz_list - label_len) # padding
            input_gaz_count[idx][flag_idx] += [0] * (max_gaz_list - label_len) # padding

            mask = [0] * label_len + [1] * (max_gaz_list - label_len)
            gaz_mask.append(mask)
        input_gaz_layer_mask.append(gaz_mask)

    assert len(np.shape(input_gaz_layer)) == 3, np.array(input_gaz_layer)
    assert len(np.shape(input_gaz_layer_mask)) == 3, np.array(input_gaz_layer_mask)
    assert not np.isnan(0 / np.sum(1 - np.array(input_gaz_layer_mask), -1)).any(), (
        np.where(0 / np.sum(1 - np.array(input_gaz_layer_mask), -1) == np.NAN), np.array(input_gaz_layer_mask))

def generate_gaz_example(words, gaz, gaz_count, gaz_alphabet):
    word_len = len(words)
    input_gaz_layer_mask = []
    input_gaz_layer = [[[] for __ in range(4)] for _ in range(word_len)]
    input_gaz_count = [[[] for __ in range(4)] for _ in range(word_len)]
    max_gaz_list = 0
    for idx in range(word_len):
        matched_list = gaz.enumerate_match_list(words[idx:])
        matched_length = [len(entity) for entity in matched_list]
        matched_ids = [gaz_alphabet.get_index(entity) for entity in matched_list]

        for w_len, matched_idx, entity in zip(matched_length, matched_ids, matched_list):
            if w_len == 1:
                input_gaz_layer[idx][3].append(matched_idx)
                input_gaz_count[idx][3].append(1)
            else:
                input_gaz_layer[idx][0].append(matched_idx)
                input_gaz_count[idx][0].append(gaz_count[matched_idx])
                input_gaz_layer[idx + w_len - 1][2].append(matched_idx)
                input_gaz_count[idx + w_len - 1][2].append(gaz_count[matched_idx])
                for cur_len in range(w_len - 2):
                    input_gaz_layer[idx + cur_len + 1][1].append(matched_idx)
                    input_gaz_count[idx + cur_len + 1][1].append(gaz_count[matched_idx])

        # if no matched entity, append zero
        for flag_idx in range(4):
            if not input_gaz_layer[idx][flag_idx]:
                input_gaz_layer[idx][flag_idx].append(0)
                input_gaz_count[idx][flag_idx].append(1)
            max_gaz_list = max(len(input_gaz_layer[idx][flag_idx]), max_gaz_list)
    pad_gaz_example(word_len, input_gaz_layer, input_gaz_layer_mask, input_gaz_count, max_gaz_list)
    return input_gaz_layer, input_gaz_layer_mask, input_gaz_count

def generate_gaz_example_by_threshold(words, gaz, gaz_count, gaz_alphabet, 
                                max_gaz_list=None, max_seq_length=None):
    if not max_gaz_list or not max_seq_length:
        print('need setting max_gaz_list and max_seq_length!')
        exit()
    
    seq_len = max_seq_length
    input_gaz_layer_mask = []
    input_gaz_layer = [[[] for __ in range(4)] for _ in range(seq_len)]
    input_gaz_count = [[[] for __ in range(4)] for _ in range(seq_len)]
    for idx in range(seq_len):
        matched_list = gaz.enumerate_match_list(words[idx: max_seq_length])
        matched_list = matched_list[: max_gaz_list]
        matched_length = [len(entity) for entity in matched_list]
        matched_ids = [gaz_alphabet.get_index(entity) for entity in matched_list]

        for w_len, matched_idx, entity in zip(matched_length, matched_ids, matched_list):
            if w_len == 1:
                input_gaz_layer[idx][3].append(matched_idx)
                input_gaz_count[idx][3].append(1)
            else:
                input_gaz_layer[idx][0].append(matched_idx)
                input_gaz_count[idx][0].append(gaz_count[matched_idx])
                input_gaz_layer[idx + w_len - 1][2].append(matched_idx)
                input_gaz_count[idx + w_len - 1][2].append(gaz_count[matched_idx])
                for cur_len in range(w_len - 2):
                    input_gaz_layer[idx + cur_len + 1][1].append(matched_idx)
                    input_gaz_count[idx + cur_len + 1][1].append(gaz_count[matched_idx])

        # if no matched entity, append zero
        for flag_idx in range(4):
            if not input_gaz_layer[idx][flag_idx]:
                input_gaz_layer[idx][flag_idx].append(0)
                input_gaz_count[idx][flag_idx].append(1)
    pad_gaz_example(seq_len, input_gaz_layer, input_gaz_layer_mask, input_gaz_count, max_gaz_list)
    return input_gaz_layer, input_gaz_layer_mask, input_gaz_count




def normalize_word(word):
    new_word = ""
    for char in word:
        if char.isdigit():
            new_word += '0'
        else:
            new_word += char
    return new_word

def load_pretrain_emb(file_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(file_path, binary=False)
    embedded_dict = {}
    for word in model.vocab:
        embedded_dict[word] = model.wv[word]
    embed_dim = model.vector_size
    return embedded_dict, embed_dim
    


def build_pretrain_embeddings(file_path, word_alphabet, embed_dim=100, norm=False):
    assert file_path, "embedding_path is None!!!"
    embedded_dict, embed_dim = load_pretrain_emb(file_path)
    scale = np.sqrt(3. / embed_dim)

    perfect_match = 0
    case_match = 0
    not_match = 0
    pretrain_emb = np.random.uniform(-scale, scale, [word_alphabet.size(), embed_dim])
    print(len(word_alphabet.instance2index))
    for word, idx in word_alphabet.instance2index.items():
        if word in embedded_dict:
            pretrain_emb[idx, :] = embedded_dict[word]
            perfect_match += 1
        elif word.lower() in embedded_dict:
            pretrain_emb[idx, :] = embedded_dict[word.lower()]
            case_match += 1
        else:
            not_match += 1
    pretrained_size = len(embedded_dict)
    print("Embeddings:")
    print("pretrain word of w2v file: %s"%(pretrained_size))
    print("prefect match: %s"%(perfect_match))
    print("case_match: %s"%(case_match))
    print("oov: %s"%(not_match))
    print("oov%%: %s"%((not_match + 0.) / word_alphabet.size()))

    return pretrain_emb, embed_dim


def build_gaz_pretrain_embeddings(gaz_file, gaz_alphabet):
    pretrain_gaz_embeddings, embed_dim = build_pretrain_embeddings(gaz_file, gaz_alphabet)
    return pretrain_gaz_embeddings


def match_all_entities(word_list, gaz, gaz_count, alphabet):
    w_len = len(word_list)
    entities = []
    for idx in range(w_len):
        matched_entities = gaz.enumerate_match_list(word_list[idx:])
        #print(matched_entities)
        entities += matched_entities
        for entity in matched_entities:
            gaz_count[entity] = 0
            alphabet.add(entity)
    return entities


def remove_covered_entities(entities, gaz_count):
    entities.sort(key=lambda x: -len(x))
    while entities:
        longest = entities[0]
        gaz_count[longest] += 1
        gaz_len = len(longest)
        for i in range(gaz_len):
            for j in range(i + 1, gaz_len + 1):
                covering_gaz = longest[i: j]
                if covering_gaz in entities:
                    entities.remove(covering_gaz)

                        






if __name__ == '__main__':
    embeddings_file = '../../embeddings/ctb.50d.vec'
    embeddings_file = '../../embeddings/gigaword_chn.all.a2b.uni.ite50.vec'
    #input_file = '../../data/train.tsv'
    input_file = 'test.tsv'
    embedding_table = build_gaz_pretrain_embeddings(embeddings_file, alphabet)
