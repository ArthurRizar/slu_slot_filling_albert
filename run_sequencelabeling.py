# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""BERT finetuning runner."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import shutil
import collections
import csv
import os
import numpy as np
import tensorflow as tf


import modeling
import optimization_finetuning as optimization
import tokenization
import conlleval
import random

# from loss import bi_tempered_logistic_loss

tf.enable_eager_execution()

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
        "data_dir", None,
        "The input data dir. Should contain the .tsv files (or other data files) "
        "for the task.")

flags.DEFINE_string(
        "bert_config_file", None,
        "The config json file corresponding to the pre-trained BERT model. "
        "This specifies the model architecture.")

flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string("vocab_file", None,
                                        "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_string(
        "output_dir", None,
        "The output directory where the model checkpoints will be written.")

## Other parameters

flags.DEFINE_string(
        "init_checkpoint", None,
        "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_bool(
        "do_lower_case", True,
        "Whether to lower case the input text. Should be True for uncased "
        "models and False for cased models.")

flags.DEFINE_integer(
        "max_seq_length", 128,
        "The maximum total input sequence length after WordPiece tokenization. "
        "Sequences longer than this will be truncated, and sequences shorter "
        "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
        "do_predict", False,
        "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_float("num_train_epochs", 3.0,
                                     "Total number of training epochs to perform.")

flags.DEFINE_float(
        "warmup_proportion", 0.1,
        "Proportion of training to perform linear learning rate warmup for. "
        "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 2000,
                                         "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                                         "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
        "tpu_name", None,
        "The Cloud TPU to use for training. This should be either the name "
        "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
        "url.")

tf.flags.DEFINE_string(
        "tpu_zone", None,
        "[Optional] GCE zone where the Cloud TPU is located in. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string(
        "gcp_project", None,
        "[Optional] Project name for the Cloud TPU-enabled project. If not "
        "specified, we will attempt to automatically detect the GCE project from "
        "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
        "num_tpu_cores", 8,
        "Only used if `use_tpu` is True. Total number of TPU cores to use.")


flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
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
        self.label = label


class PaddingInputExample(object):
    """Fake example so the num input examples is a multiple of the batch size.
    When running eval/predict on the TPU, we need to pad the number of examples
    to be a multiple of the batch size, because the TPU requires a fixed batch
    size. The alternative is to drop the last batch, which is bad because it means
    the entire output data won't be generated.
    We use this class instead of `None` because treating `None` as padding
    battches could cause silent errors.
    """


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self,
                             input_ids,
                             input_mask,
                             segment_ids,
                             label_id,
                             is_real_example=True):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
        self.is_real_example = is_real_example


class DataProcessor(object):
    """Base class for data converters for sequence classification data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for prediction."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()

    @classmethod
    def _read_data(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        with tf.gfile.Open(input_file, "r") as f:
            reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
            lines = []
            words = []
            labels = []
            for line in reader:
                if len(line) == 0:
                    l = ' '.join([label for label in labels if len(label) > 0])
                    w = ' '.join([word for word in words if len(word) > 0])
                    lines.append((w, l))
                    words = []
                    labels = []
                else:
                    word = line[0].strip()
                    label = line[1].strip()
                    words.append(word)
                    labels.append(label)
            return lines


class TestProcessor(DataProcessor):
    def get_train_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_data(os.path.join(data_dir, "train.tsv")), "train")

    def get_dev_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_data(os.path.join(data_dir, "dev.tsv")), "dev")

    def get_test_examples(self, data_dir):
        """See base class."""
        return self._create_examples(
                self._read_data(os.path.join(data_dir, "test.tsv")), "test")

    def get_labels(self):
        with codecs.open('data/labels.tsv', 'r', 'utf8') as fr:
            labels = []
            for line in fr:
                line = line.strip().upper()
                labels.append(line)
        return labels

    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev sets."""
        examples = []
        for (i, line) in enumerate(lines):
            # Only the test set has a header
            guid = "%s-%d" % (set_type, i)
            text_a = tokenization.convert_to_unicode(line[0]).lower()
            label = tokenization.convert_to_unicode(line[1]).upper()
            examples.append(
                    InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples

def convert_single_example(ex_index, example, label_map, max_seq_length,
                                                     tokenizer):
    """Converts a single `InputExample` into a single `InputFeatures`."""
    text_list = example.text_a.split(' ')
    label_list = example.label.split(' ')
    tokens = []
    labels = []
    for (word, label) in zip(text_list, label_list):
        token = tokenizer.tokenize(word)
        tokens.extend(token)
        for i, _ in enumerate(token):
            if i == 0:
                labels.append(label)
            else:
                labels.append('[WordPiece]')

    if len(tokens) > max_seq_length - 2:
        tokens = tokens[: max_seq_length - 2]
        labels = labels[: max_seq_length - 2]

    final_tokens = []
    segment_ids = []
    label_ids = []

    final_tokens.append("[CLS]")
    label_ids.append(label_map['[CLS]'])
    #label_ids.append(label_map['O')
    segment_ids.append(0)
    for token, label in zip(tokens, labels):
        final_tokens.append(token)
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
        #label_ids.append(0)
        label_ids.append(label_map['[PAD]'])
        final_tokens.append('[PAD]')

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length
    assert len(label_ids) == max_seq_length
    assert len(final_tokens) == max_seq_length

    #print(example.label)
    if ex_index < 3:
        tf.logging.info("*** Example ***")
        tf.logging.info("guid: %s" % (example.guid))
        tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
        tf.logging.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
        tf.logging.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
        tf.logging.info("segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
        tf.logging.info("label_ids: %s " % " ".join([str(x) for x in label_ids]))

    feature = InputFeatures(
            input_ids=input_ids,
            input_mask=input_mask,
            segment_ids=segment_ids,
            label_id=label_ids,
            is_real_example=True)
    return feature




def file_based_convert_examples_to_features(
        examples, label_map, max_seq_length, tokenizer, output_file):
    """Convert a set of `InputExample`s to a TFRecord file."""

    writer = tf.python_io.TFRecordWriter(output_file)

    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_map,
                                max_seq_length, tokenizer)

        def create_int_feature(values):
            f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return f

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(feature.input_ids)
        features["input_mask"] = create_int_feature(feature.input_mask)
        features["segment_ids"] = create_int_feature(feature.segment_ids)
        features["label_ids"] = create_int_feature(feature.label_id)
        features["is_real_example"] = create_int_feature(
                [int(feature.is_real_example)])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))
        writer.write(tf_example.SerializeToString())
    writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                                                drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
    }

    def _decode_record(record, name_to_features):
        """Decodes a record to a TensorFlow example."""
        example = tf.parse_single_example(record, name_to_features)

        # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
        # So cast all int64 to int32.
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t

        return example

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        # For training, we want a lot of parallel reading and shuffling.
        # For eval, we want no shuffling and parallel reading doesn't matter.
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.apply(
                tf.contrib.data.map_and_batch(
                        lambda record: _decode_record(record, name_to_features),
                        batch_size=batch_size,
                        drop_remainder=drop_remainder))

        return d

    return input_fn


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


def create_model(bert_config, is_training, input_ids, input_mask, segment_ids,
                                 labels, num_labels, use_one_hot_embeddings):
    """Creates a classification model."""
    model = modeling.BertModel(
            config=bert_config,
            is_training=is_training,
            input_ids=input_ids,
            input_mask=input_mask,
            token_type_ids=segment_ids,
            use_one_hot_embeddings=use_one_hot_embeddings)

    # In the demo, we are doing a simple classification task on the entire
    # segment.
    #
    # If you want to use the token-level output, use model.get_sequence_output()
    # instead.

    '''
    output_layer = model.get_pooled_output()

    hidden_size = output_layer.shape[-1].value

    with tf.variable_scope("loss"):
        ln_type = bert_config.ln_type
        if ln_type == 'preln': 
            # add by brightmart, 10-06. if it is preln, we need to an additonal layer: layer normalization as suggested 
            # in paper "ON LAYER NORMALIZATION IN THE TRANSFORMER ARCHITECTURE"
            print("ln_type is preln. add LN layer.")
            output_layer=layer_norm(output_layer)
        else:
            print("ln_type is postln or other,do nothing.")
    '''

    loss, per_example_loss, logits, acc, pred_label_ids, probabilities = inference(model, num_labels, is_training, labels, input_mask)

    return (loss, per_example_loss, logits, pred_label_ids, probabilities)

def project_layer(inputs, num_labels):
    #logits = tf.layers.dense(inputs, num_labels, activation=None, kernel_initializer=tf.truncated_normal_initializer(stddev=0.02))
    logits = tf.layers.dense(inputs, num_labels, activation=None)
    logits = tf.identity(logits, name='logits')
    return logits


def birnn_layer(inputs, num_units=768, cell_type='lstm', num_layers=1, dropout_keep_prob=None):
    with tf.variable_scope('birnn_layers'):
        if cell_type == 'lstm':
            cell = tf.contrib.rnn.LSTMCell(num_units)
        else:
            cell = tf.contrib.rnn.GRUCell(num_units)
        cell_fw = cell
        cell_bw = cell

        if dropout_keep_prob is not None:
           cell_fw = tf.contrib.rnn.DropoutWrapper(cell_fw, output_keep_prob=dropout_keep_prob)
           cell_bw = tf.contrib.rnn.DropoutWrapper(cell_bw, output_keep_prob=dropout_keep_prob)

        if num_layers > 1:
           cell_fw = tf.contrib.rnn.MultiRNNCell([cell_fw]*num_layers, state_is_tuple=True)
           cell_bw = tf.contrib.rnn.MultiRNNCell([cell_bw]*num_layers, state_is_tuple=True)

        outputs, _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs,
                                                        dtype=tf.float32)

    outputs = tf.concat(outputs, axis=2)
    return outputs

def inference(model, num_labels, is_training, labels, input_mask, use_crf=True, use_rnn=False):
    output_layer = model.get_sequence_output()
    hidden_size = output_layer.shape[-1].value
    if is_training:
        keep_prob = 0.9
    else:
        keep_prob = 1.0

    #use_rnn = True
    if use_rnn:
        output_layer = birnn_layer(output_layer, num_units=768, dropout_keep_prob=keep_prob)

    #projection
    logits = project_layer(output_layer, num_labels)

    one_hot_labels = tf.one_hot(labels, depth=num_labels, dtype=tf.float32)
    #use_crf = False
    with tf.variable_scope("loss"):
        if use_crf:
            mask2len = tf.reduce_sum(input_mask, axis=1)
            log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(
                                                        inputs=logits,
                                                        tag_indices=labels,
                                                        sequence_lengths=mask2len)

            per_example_loss = -log_likelihood
            loss = tf.reduce_mean(-log_likelihood)
            pred_label_ids, viterbi_score = tf.contrib.crf.crf_decode(logits, transition_params, mask2len)
            pred_label_ids = tf.identity(pred_label_ids, name='crf_pred_label_ids')
            probabilities = tf.identity(viterbi_score, name='crf_probs')
        else:
            losses = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=one_hot_labels)
            per_example_loss = losses
            mask = tf.cast(input_mask, dtype=tf.float32)
            loss = losses * mask

            loss = tf.reduce_sum(loss)

            total_size = tf.reduce_sum(mask) + 1e-12
            loss /= total_size


            probabilities = tf.nn.softmax(logits, axis=-1, name='probs')
            pred_label_ids = tf.argmax(probabilities, axis=-1, name='pred_label_ids')
            pred_label_ids = tf.cast(pred_label_ids, dtype=tf.int32)


        correct_pred = tf.equal(labels, pred_label_ids)
        acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='acc')
        loss = tf.identity(loss, name='loss')

    return (loss, per_example_loss, logits, acc, pred_label_ids, probabilities)


def layer_norm(input_tensor, name=None):
    """Run layer normalization on the last dimension of the tensor."""
    return tf.contrib.layers.layer_norm(
            inputs=input_tensor, begin_norm_axis=-1, begin_params_axis=-1, scope=name)

def model_fn_builder(bert_config, num_labels, init_checkpoint, learning_rate,
                                         num_train_steps, num_warmup_steps, use_tpu,
                                         use_one_hot_embeddings):
    """Returns `model_fn` closure for TPUEstimator."""

    def model_fn(features, labels, mode, params):    # pylint: disable=unused-argument
        """The `model_fn` for TPUEstimator."""

        tf.logging.info("*** Features ***")
        for name in sorted(features.keys()):
            tf.logging.info("    name = %s, shape = %s" % (name, features[name].shape))

        input_ids = features["input_ids"]
        input_mask = features["input_mask"]
        segment_ids = features["segment_ids"]
        label_ids = features["label_ids"]
        is_real_example = None
        if "is_real_example" in features:
            is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)
        else:
            is_real_example = tf.ones(tf.shape(label_ids), dtype=tf.float32)

        input_ids = tf.placeholder_with_default(input_ids, shape=[None, input_ids.shape[1]], name='input_ids')
        input_mask = tf.placeholder_with_default(input_mask, shape=[None, input_mask.shape[1]], name='input_mask')
        segment_ids = tf.placeholder_with_default(segment_ids, shape=[None, segment_ids.shape[1]], name='segment_ids')
        

        is_training = (mode == tf.estimator.ModeKeys.TRAIN)

        (total_loss, per_example_loss, logits, pred_label_ids, probabilities) = create_model(
                bert_config, is_training, input_ids, input_mask, segment_ids, label_ids,
                num_labels, use_one_hot_embeddings)

        tvars = tf.trainable_variables()
        initialized_variable_names = {}
        scaffold_fn = None
        if init_checkpoint:
            (assignment_map, initialized_variable_names
            ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
            if use_tpu:

                def tpu_scaffold():
                    tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
                    return tf.train.Scaffold()

                scaffold_fn = tpu_scaffold
            else:
                tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

        tf.logging.info("**** Trainable Variables ****")
        for var in tvars:
            init_string = ""
            if var.name in initialized_variable_names:
                init_string = ", *INIT_FROM_CKPT*"
            tf.logging.info("    name = %s, shape = %s%s", var.name, var.shape,
                                            init_string)

        output_spec = None
        if mode == tf.estimator.ModeKeys.TRAIN:

            train_op = optimization.create_optimizer(
                    total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    train_op=train_op,
                    scaffold_fn=scaffold_fn)
        elif mode == tf.estimator.ModeKeys.EVAL:

            def metric_fn(per_example_loss, label_ids, logits, is_real_example):
                predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
                accuracy = tf.metrics.accuracy(
                        labels=label_ids, predictions=predictions, weights=is_real_example)
                loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
                return {
                        "eval_accuracy": accuracy,
                        "eval_loss": loss,
                }
            eval_metrics = (metric_fn,
                                [per_example_loss, label_ids, logits, is_real_example])
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    loss=total_loss,
                    eval_metrics=eval_metrics,
                    scaffold_fn=scaffold_fn)
        else:
            output_spec = tf.contrib.tpu.TPUEstimatorSpec(
                    mode=mode,
                    predictions={"pred_label_ids": pred_label_ids},
                    scaffold_fn=scaffold_fn)
        return output_spec

    return model_fn


# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def input_fn_builder(features, seq_length, is_training, drop_remainder):
    """Creates an `input_fn` closure to be passed to TPUEstimator."""

    all_input_ids = []
    all_input_mask = []
    all_segment_ids = []
    all_label_ids = []

    for feature in features:
        all_input_ids.append(feature.input_ids)
        all_input_mask.append(feature.input_mask)
        all_segment_ids.append(feature.segment_ids)
        all_label_ids.append(feature.label_id)

    def input_fn(params):
        """The actual input function."""
        batch_size = params["batch_size"]

        num_examples = len(features)

        # This is for demo purposes and does NOT scale to large data sets. We do
        # not use Dataset.from_generator() because that uses tf.py_func which is
        # not TPU compatible. The right way to load data is with TFRecordReader.
        d = tf.data.Dataset.from_tensor_slices({
                "input_ids":
                        tf.constant(
                                all_input_ids, shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "input_mask":
                        tf.constant(
                                all_input_mask,
                                shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "segment_ids":
                        tf.constant(
                                all_segment_ids,
                                shape=[num_examples, seq_length],
                                dtype=tf.int32),
                "label_ids":
                        tf.constant(all_label_ids, shape=[num_examples, seq_length], dtype=tf.int32),
        })

        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=100)

        d = d.batch(batch_size=batch_size, drop_remainder=drop_remainder)
        return d

    return input_fn



# This function is not used by this file but is still used by the Colab and
# people who depend on it.
def convert_examples_to_features(examples, label_list, max_seq_length,
                                                                 tokenizer):
    """Convert a set of `InputExample`s to a list of `InputFeatures`."""

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 10000 == 0:
            tf.logging.info("Writing example %d of %d" % (ex_index, len(examples)))

        feature = convert_single_example(ex_index, example, label_list,
                                                                         max_seq_length, tokenizer)

        features.append(feature)
    return features


def print_metrics(label_map, record_file, tokenizer, predictions, num_features, seq_length):
    def _parse_function(exam_proto):
        name_to_features = {
            "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
            "is_real_example": tf.FixedLenFeature([], tf.int64),
            } 
        return tf.io.parse_single_example(exam_proto, name_to_features)
        
    reader = tf.data.TFRecordDataset(record_file)
    reader = reader.map(_parse_function)
    iterator = reader.batch(1) 
    iterator = iterator.take(num_features)
 
    features = []
    for item in iterator:
        for key in item:
            item[key] = np.array(item[key][0])
        features.append(item)
    id2label = {label_map[label]: label for label in label_map}

    output_predict_file = os.path.join(FLAGS.output_dir, 'predict.result')
    with codecs.open(output_predict_file, 'w', 'utf8') as fw:
        for feature, prediction in zip(features, predictions):
            token_ids = feature['input_ids']
            label_ids = feature['label_ids']
            tokens = tokenizer.convert_ids_to_tokens(token_ids)
            line = ''
            predict_label_ids = prediction['pred_label_ids']
            for token, label_id, pred_label_id in zip(tokens, label_ids, predict_label_ids):
                pred_label = id2label[pred_label_id]
                label = id2label[label_id]
                if pred_label in ['[CLS]', '[SEP]'] or token == '[PAD]':
                    continue
                line += token + '\t' + label + '\t' + pred_label + '\n'
            fw.write(line + '\n')

    eval_result = conlleval.return_report(output_predict_file)
    print(''.join(eval_result))
    overall, _ = conlleval.metrics(conlleval.evaluate(codecs.open(output_predict_file, 'r', 'utf8')))
    mean_f1 = overall.fscore
    print('mean_f1:', mean_f1)
    return eval_result

def get_and_save_label_map(label_list, output_dir):
    label_map = {}
    with codecs.open(output_dir + '/label_map', 'w', 'utf8') as fw:
        for (i, label) in enumerate(label_list):
            label_map[label] = i
            fw.write(str(i) + '\t' + label + '\n')
    return label_map


def main(_):
    tf.logging.set_verbosity(tf.logging.INFO)
    rng = random.Random(FLAGS.random_seed)
    processors = {
            "ner": TestProcessor
    }

    tokenization.validate_case_matches_checkpoint(FLAGS.do_lower_case,
                                                                                                FLAGS.init_checkpoint)

    if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict:
        raise ValueError(
                "At least one of `do_train`, `do_eval` or `do_predict' must be True.")

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    if FLAGS.max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (FLAGS.max_seq_length, bert_config.max_position_embeddings))

    tf.gfile.MakeDirs(FLAGS.output_dir)

    task_name = FLAGS.task_name.lower()

    if task_name not in processors:
        raise ValueError("Task not found: %s" % (task_name))

    processor = processors[task_name]()

    label_list = processor.get_labels()
    
    shutil.copy(FLAGS.vocab_file, FLAGS.output_dir)
    shutil.copy(FLAGS.bert_config_file, FLAGS.output_dir)

    tokenizer = tokenization.FullTokenizer(
            vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

    tpu_cluster_resolver = None
    if FLAGS.use_tpu and FLAGS.tpu_name:
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

    is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
    # Cloud TPU: Invalid TPU configuration, ensure ClusterResolver is passed to tpu.
    print("###tpu_cluster_resolver:",tpu_cluster_resolver)
    model_output_dir = os.path.join(FLAGS.output_dir, 'checkpoints')
    run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            master=FLAGS.master,
            model_dir=model_output_dir,
            save_checkpoints_steps=FLAGS.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                    iterations_per_loop=FLAGS.iterations_per_loop,
                    num_shards=FLAGS.num_tpu_cores,
                    per_host_input_for_training=is_per_host))

    train_examples = None
    num_train_steps = None
    num_warmup_steps = None
    if FLAGS.do_train:
        train_examples = processor.get_train_examples(FLAGS.data_dir) # TODO
        rng.shuffle(train_examples)
        print("###length of total train_examples:",len(train_examples))
        num_train_steps = int(len(train_examples)/ FLAGS.train_batch_size * FLAGS.num_train_epochs)
        num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)


    label_list = processor.get_labels()
    label_map = get_and_save_label_map(label_list, FLAGS.output_dir)


    model_fn = model_fn_builder(
            bert_config=bert_config,
            num_labels=len(label_list),
            init_checkpoint=FLAGS.init_checkpoint,
            learning_rate=FLAGS.learning_rate,
            num_train_steps=num_train_steps,
            num_warmup_steps=num_warmup_steps,
            use_tpu=FLAGS.use_tpu,
            use_one_hot_embeddings=FLAGS.use_tpu)

    # If TPU is not available, this will fall back to normal Estimator on CPU
    # or GPU.
    estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=FLAGS.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=FLAGS.train_batch_size,
            eval_batch_size=FLAGS.eval_batch_size,
            predict_batch_size=FLAGS.predict_batch_size)

    if FLAGS.do_train:
        train_file = os.path.join(FLAGS.output_dir, "train.tf_record")
        train_file_exists = os.path.exists(train_file)
        print("###train_file_exists:", train_file_exists," ;train_file:", train_file)
        if not train_file_exists: # if tf_record file not exist, convert from raw text file. # TODO
                file_based_convert_examples_to_features(train_examples, label_map, FLAGS.max_seq_length, tokenizer, train_file)
        tf.logging.info("***** Running training *****")
        tf.logging.info("    Num examples = %d", len(train_examples))
        tf.logging.info("    Batch size = %d", FLAGS.train_batch_size)
        tf.logging.info("    Num steps = %d", num_train_steps)
        train_input_fn = file_based_input_fn_builder(
                input_file=train_file,
                seq_length=FLAGS.max_seq_length,
                is_training=True,
                drop_remainder=True)
        estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

    if FLAGS.do_eval:
        eval_examples = processor.get_dev_examples(FLAGS.data_dir)
        num_actual_eval_examples = len(eval_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on. These do NOT count towards the metric (all tf.metrics
            # support a per-instance weight, and these get a weight of 0.0).
            while len(eval_examples) % FLAGS.eval_batch_size != 0:
                eval_examples.append(PaddingInputExample())

        eval_file = os.path.join(FLAGS.output_dir, "eval.tf_record")
        file_based_convert_examples_to_features(
                eval_examples, label_map, FLAGS.max_seq_length, tokenizer, eval_file)

        tf.logging.info("***** Running evaluation *****")
        tf.logging.info("    Num examples = %d (%d actual, %d padding)",
                                        len(eval_examples), num_actual_eval_examples,
                                        len(eval_examples) - num_actual_eval_examples)
        tf.logging.info("    Batch size = %d", FLAGS.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        if FLAGS.use_tpu:
            assert len(eval_examples) % FLAGS.eval_batch_size == 0
            eval_steps = int(len(eval_examples) // FLAGS.eval_batch_size)

        eval_drop_remainder = True if FLAGS.use_tpu else False
        eval_input_fn = file_based_input_fn_builder(
                input_file=eval_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=eval_drop_remainder)

        #######################################################################################################################
        # evaluate all checkpoints; you can use the checkpoint with the best dev accuarcy
        steps_and_files = []
        filenames = tf.gfile.ListDirectory(model_output_dir)
        for filename in filenames:
            if filename.endswith(".index"):
                ckpt_name = filename[:-6]
                cur_filename = os.path.join(model_output_dir, ckpt_name)
                global_step = int(cur_filename.split("-")[-1])
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files.append([global_step, cur_filename])
        steps_and_files = sorted(steps_and_files, key=lambda x: x[0])


        output_eval_file = os.path.join(FLAGS.output_dir, "eval_results_albert_zh.txt")
        print("output_eval_file:", output_eval_file)
        tf.logging.info("output_eval_file:" + output_eval_file)
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            for global_step, filename in sorted(steps_and_files, key=lambda x: x[0]):
                #result = estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps, checkpoint_path=filename)

                #tf.logging.info("***** Eval results %s *****" % (filename))
                #writer.write("***** Eval results %s *****\n" % (filename))
                #for key in sorted(result.keys()):
                #    if key == 'eval_predictions':
                #        continue
                #    tf.logging.info("    %s = %s", key, str(result[key]))
                #    writer.write("%s = %s\n" % (key, str(result[key])))


                tf.logging.info("***** Eval results %s *****" % (filename))
                writer.write("***** Eval results %s *****\n" % (filename))
                result = estimator.predict(input_fn=eval_input_fn)
                metrics = print_metrics(label_map, eval_file, tokenizer, result, num_actual_eval_examples, FLAGS.max_seq_length)
                writer.write(''.join(metrics))

    if FLAGS.do_predict:
        predict_examples = processor.get_test_examples(FLAGS.data_dir)
        num_actual_predict_examples = len(predict_examples)
        if FLAGS.use_tpu:
            # TPU requires a fixed batch size for all batches, therefore the number
            # of examples must be a multiple of the batch size, or else examples
            # will get dropped. So we pad with fake examples which are ignored
            # later on.
            while len(predict_examples) % FLAGS.predict_batch_size != 0:
                predict_examples.append(PaddingInputExample())

        predict_file = os.path.join(FLAGS.output_dir, "predict.tf_record")
        file_based_convert_examples_to_features(predict_examples, label_map,
                                                                FLAGS.max_seq_length, tokenizer,
                                                                predict_file)

        tf.logging.info("***** Running prediction*****")
        tf.logging.info("    Num examples = %d (%d actual, %d padding)",
                                        len(predict_examples), num_actual_predict_examples,
                                        len(predict_examples) - num_actual_predict_examples)
        tf.logging.info("    Batch size = %d", FLAGS.predict_batch_size)

        predict_drop_remainder = True if FLAGS.use_tpu else False
        predict_input_fn = file_based_input_fn_builder(
                input_file=predict_file,
                seq_length=FLAGS.max_seq_length,
                is_training=False,
                drop_remainder=predict_drop_remainder)

        result = estimator.predict(input_fn=predict_input_fn)

        output_predict_file = os.path.join(FLAGS.output_dir, "test_results.tsv")
        tf.logging.info("output_predict_file:" + output_predict_file)
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            '''
            num_written_lines = 0
            tf.logging.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                pred_label_ids = prediction["pred_label_ids"]
                if i >= num_actual_predict_examples:
                    break
                output_line = "\t".join(
                        str(pred_label_id)
                        for pred_label_id in pred_label_ids) + "\n"
                writer.write(output_line)
                num_written_lines += 1
            '''
            metrics = print_metrics(label_map, predict_file, tokenizer, result, num_actual_predict_examples, FLAGS.max_seq_length)
            writer.write(''.join(metrics))



if __name__ == "__main__":
    flags.mark_flag_as_required("data_dir")
    flags.mark_flag_as_required("task_name")
    flags.mark_flag_as_required("vocab_file")
    flags.mark_flag_as_required("bert_config_file")
    flags.mark_flag_as_required("output_dir")
    tf.app.run()
