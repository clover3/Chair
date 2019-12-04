import collections
import os
import pickle
import sys

import numpy as np

from data_generator import tokenizer_wo_tf as tokenization
from data_generator.common import get_tokenizer
from misc_lib import TimeEstimator
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature, create_float_feature
from tlm.tf_logging import tf_logging, ab_logging


class PredictionOutput:
    def __init__(self, path):
        data = pickle.load(open(path, "rb"))
        input_ids = []
        masked_lm_example_loss = []
        masked_lm_positions = []
        masked_lm_ids = []
        batch_size, seq_length = data[0]['masked_input_ids'].shape
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.prediction_per_seq = int(data[0]['masked_lm_example_loss'].shape[0] / batch_size)

        for e in data:
            input_ids.append(e["input_ids"])
            losses = np.reshape(e["masked_lm_example_loss"], [-1, self.prediction_per_seq])
            assert len(losses) == len(e["input_ids"])
            masked_lm_example_loss.append(losses)
            masked_lm_positions.append(e["masked_lm_positions"])
            masked_lm_ids.append(e["masked_lm_ids"])

        self.input_ids = np.concatenate(input_ids)
        self.masked_lm_example_loss = np.concatenate(masked_lm_example_loss)
        self.masked_lm_positions = np.concatenate(masked_lm_positions)
        self.masked_lm_ids = np.concatenate(masked_lm_ids)
        self.masked_lm_weights = (np.not_equal(self.masked_lm_ids, 0)).astype(float)


def assert_input_equal(input_ids1, input_ids2):
    assert input_ids1.shape == input_ids2.shape
    for i in range(len(input_ids1)):
        assert input_ids1[i] == input_ids2[i]




def get_segment_and_mask(input_ids, sep_id):
    indice = np.where(input_ids == sep_id)[0]
    if len(indice) != 2:
        raise Exception()
    return get_segment_and_mask(input_ids, indice)


def get_segment_and_mask_inner(input_ids, sep_indice):
    a_len = sep_indice[0]+1
    b_len = sep_indice[1]+1 - a_len
    pad_len = len(input_ids) - (a_len + b_len)
    segment_ids = [0] * a_len + [1] * b_len + [0] * pad_len
    input_mask = [1] * (a_len + b_len) + [0] * pad_len
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    return features


def get_sep_considering_masking(input_ids, sep_id, masked_lm_ids, masked_lm_positions):
    location_to_id = {k: v for k, v in zip(masked_lm_positions, masked_lm_ids)}
    valid_sep = []
    for i in range(len(input_ids)):
        if input_ids[i] == sep_id:
            if i in location_to_id:
                pass
            else:
                valid_sep.append(i)

    if len(valid_sep) != 2 :
        raise Exception()
    return valid_sep







def do(data_id):
    tokenzier = get_tokenizer()
    name1 = os.path.join("disk_output","bert_{}.pickle".format(data_id))
    name2 = os.path.join("disk_output", "bert_{}.pickle".format(data_id))

    tf_logging.debug("Loading data 1")
    output1 = PredictionOutput(name1)
    tf_logging.debug("Loading data 2")
    output2 = PredictionOutput(name2)

    assert len(output1.input_ids) == len(output2.input_ids)

    record_writer = RecordWriterWrap("disk_output/loss_pred_{}".format(data_id))
    n_inst = len(output1.input_ids)
    sep_id = tokenzier.vocab["[SEP]"]
    tf_logging.debug("Iterating")
    ticker = TimeEstimator(n_inst, "", 1000)
    for i in range(n_inst):
        if i % 1000 == 0:
            tf_logging.debug("Iterating {}".format(i))
            assert_input_equal(output1.input_ids[i], output2.input_ids[i])
        try:
            features = get_segment_and_mask(output1.input_ids[i], sep_id)
        except:
            try:
                sep_indice = get_sep_considering_masking(output1.input_ids[i], sep_id, output1.masked_lm_ids[i], output1.masked_lm_positions[i])
                features = get_segment_and_mask_inner(output1.input_ids[i], sep_indice)
            except:
                tokens = tokenzier.convert_ids_to_tokens(output1.input_ids[i])
                print(tokenization.pretty_tokens(tokens))
                print(output1.masked_lm_ids[i])
                print(output1.masked_lm_positions[i])
                raise

        features["next_sentence_labels"] = create_int_feature([0])
        features["masked_lm_positions"] = create_int_feature(output1.masked_lm_positions[i])
        features["masked_lm_ids"] = create_int_feature(output1.masked_lm_ids[i])
        features["masked_lm_weights"] = create_float_feature(output1.masked_lm_weights[i])
        features["loss_base"] = create_float_feature(output1.masked_lm_example_loss[i])
        features["loss_target"] = create_float_feature(output2.masked_lm_example_loss[i])
        record_writer.write_feature(features)
        ticker.tick()

    record_writer.close()


if __name__ == '__main__':
    tf_logging.setLevel(ab_logging.DEBUG)
    do(sys.argv[1])


