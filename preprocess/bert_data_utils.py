#coding:utf-8
###################################################
# File Name: bert_data_utils.py
# Author: Meng Zhao
# mail: @
# Created Time: 2019年03月06日 星期三 14时04分26秒
#=============================================================
import sys
import codecs
import copy
import numpy as np

sys.path.append('../')

from preprocess import tokenization


class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, text_c=None, label=None, pool=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
                sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
                Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
                specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.text_c = text_c
        self.label = label
        self.pool = pool

class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                 input_ids,
                 input_mask,
                 segment_ids,
                 label_id=None,
                 is_real_example=True):

        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


def read_code_file(code_file):
    #get real label
    label2code = {}
    with codecs.open(code_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip()
            line_info = line.split('\t')
            label = line_info[0].lower()
            code_value = line_info[1].lower()
            label2code[label] = code_value
    return label2code


def read_bert_labels_file(label_file):
    label_list = []
    with codecs.open(label_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().lower()
            label_list.append(line)

    label_list = list(set(label_list))
    label2idx = {}
    idx2label = {}
    for (i, label) in enumerate(label_list):
        label2idx[label] = i
        idx2label[i] = label
        print(i, label)
    return label2idx, idx2label


def read_ner_label_map_file(label_map_file):
    label_map = {}
    idx2label = {}
    with codecs.open(label_map_file, 'r', 'utf8') as fr:
        for line in fr:
            line = line.strip().upper()
            line_info = line.split('\t')
            idx = int(line_info[0])
            label = line_info[1]
            label_map[label] = idx
            idx2label[idx] = label
    return label_map, idx2label 

def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()


def convert_text_to_tokens(text, max_seq_length, tokenizer):
    sizes = []
    tokens = []
    org_char = []
    for index, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        sizes.append(len(token))

    print(tokens)

def convert_online_example(example, max_seq_length, tokenizer):
    text_list = example.text_a
    tokens = []
    for index, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)

    if len(tokens) > max_seq_length - 2:
        tokens = tokens[: max_seq_length - 2]

    final_tokens = []
    segment_ids = []

    final_tokens.append("[CLS]")
    segment_ids.append(0)
    for index, token in enumerate(tokens):
        final_tokens.append(token)
        if example.label:
            label = labels[index]
        segment_ids.append(0)
    final_tokens.append("[SEP]")
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(final_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        final_tokens.append('[PAD]')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(final_tokens) == max_seq_length


    print("*** Example ***")
    print("guid: %s" % (example.guid))
    print("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
    print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
    print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
    print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))

    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            is_real_example=True)
    return feature


def convert_single_example(ex_index, example, label_map, max_seq_length, tokenizer, is_predict=True):
    print(label_map)

    text_list = example.text_a.split(' ')
    if example.label:
        label_list = example.label.split(' ')
    
    tokens = []
    labels = []
    for index, word in enumerate(text_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        
        if example.label:
            label = labes_list[index]
            for i, _ in enumerate(token):
                if i == 0:
                    labels.append(label)
                else:
                    labels.append('[WordPiece]')

    print(tokens)
    if len(tokens) > max_seq_length - 2:
        tokens = tokens[: max_seq_length - 2]

    if example.label and len(tokens) > max_seq_length - 2:
        labels = labels[: max_seq_length - 2]

    final_tokens = []
    segment_ids = []
    label_ids = []

    final_tokens.append("[CLS]")
    label_ids.append(label_map['[CLS]'])
    segment_ids.append(0)
    for index, token in enumerate(tokens):
        final_tokens.append(token)
        if example.label:
            label = labels[index]
            label_ids.append(label_map[label])
        segment_ids.append(0)
    final_tokens.append("[SEP]")
    label_ids.append(label_map['[SEP]'])
    #label_ids.append(label_map['O'])
    segment_ids.append(0)

    input_ids = tokenizer.convert_tokens_to_ids(final_tokens)

    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask = [1] * len(input_ids)

    # Zero-pad up to the sequence length.
    while len(input_ids) < max_seq_length:
        input_ids.append(0)
        input_mask.append(0)
        segment_ids.append(0)
        if example.label:
            label_ids.append(label_map['[PAD]'])
        final_tokens.append('[PAD]')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(final_tokens) == max_seq_length

    if example.label:
        assert len(label_ids) == max_seq_length

    #print(example.label)
    if ex_index < 3:
        print("*** Example ***")
        print("guid: %s" % (example.guid))
        print("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
        print("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        print("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        print("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        print("label_ids: %s " % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_ids,
            is_real_example=True)
    return feature







def get_data_from_file(file_name):
    text_trunk = []
    with codecs.open(file_name, 'r', 'utf8') as fr:
        for i, line in enumerate(fr):
            line = line.strip().lower()
            line_info = line.split('\t')
            text_a = line_info[0].strip()
            label = line_info[1].strip()
            yield InputExample(guid=i, text_a=text_a, label=label)



def file_based_convert_examples_to_features(file_name, label_map, max_seq_length, tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    examples = get_data_from_file(file_name)
    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            print("Writing example %d" % (ex_index))

        feature = convert_single_example(ex_index, example, label_map,
                                         max_seq_length, tokenizer)

        features.append(feature)
    return features




def file_based_convert_examples_to_features_with_candidates(file_name, label_map, max_seq_length, tokenizer, candidates):
    raw_examples = get_data_from_file(file_name)
    raw_examples = list(raw_examples)
    if candidates is None:
        candidates = [raw_example.text_b for raw_example in raw_examples]
        candidates = list(set(candidates))
    candidate_id_map = {v: k for k, v in enumerate(candidates)}
    #print(candidate_id_map)
    #exit()

    pool_size = len(candidates)
    print('raw examples:', len(raw_examples))
    print('candidates:', pool_size)

    candidate_input_features = []
    for i, item in enumerate(candidates):
        example = InputExample(i, text_a=item)
        feature = convert_single_example(i, example, label_map, max_seq_length, tokenizer)
        candidate_input_features.append(feature)

    features = []
    for i, raw_example in enumerate(raw_examples):
        example = InputExample(i, text_a=raw_example.text_a, label=raw_example.label)
        feature = convert_single_example(i, example, label_map, max_seq_length, tokenizer)
        features.append(feature)
    print('total number of features:', len(features))
    return features, candidate_input_features

