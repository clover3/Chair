import collections
import os
import pickle

import tensorflow as tf

from cpath import output_path
from data_generator.common import get_tokenizer
from tf_util.enum_features import load_record_v2
from tlm.dictionary.feature_to_text import take, Feature2Text
from tlm.wiki import bert_training_data as btd
from visualize.html_visual import Cell, HtmlVisualizer

working_path ="/mnt/nfs/work3/youngwookim/data/bert_tf"
if os.name == "nt":
    working_path = output_path


def visualize_prediction_data(data_id):
    tokenizer = get_tokenizer()
    num_samples_list = open(os.path.join(working_path, "entry_prediction_n", data_id), "r").readlines()
    p = os.path.join(working_path, "entry_loss", "entry{}.pickle".format(data_id))
    loss_outputs_list = pickle.load(open(p, "rb"))
    print("Loaded input data")
    loss_outputs = []
    for e in loss_outputs_list:
        loss_outputs.extend(e["masked_lm_example_loss"])
    print("Total of {} loss outputs".format(len(loss_outputs)))
    instance_idx = 0
    feature_itr = load_record_v2(os.path.join(working_path, "entry_prediction_tf.done", data_id))
    n = len(num_samples_list)
    n = 100
    html = HtmlVisualizer("entry_prediction.html")
    for i in range(n):
        n_sample = int(num_samples_list[i])
        assert n_sample > 0
        first_inst = feature_itr.__next__()
        feature = Feature2Text(first_inst, tokenizer)

        html.write_headline("Input:")
        html.write_paragraph(feature.get_input_as_text(True, True))
        html.write_headline("Word:" + feature.get_selected_word_text())

        if instance_idx + n_sample >= len(loss_outputs):
            break

        if n_sample == 1:
            continue

        rows = []
        no_dict_loss = loss_outputs[instance_idx]
        row = [Cell(no_dict_loss, 0), Cell("")]
        rows.append(row)
        instance_idx += 1
        for j in range(1, n_sample):
            feature = Feature2Text(feature_itr.__next__(), tokenizer)
            def_cell = Cell(feature.get_def_as_text())
            loss = loss_outputs[instance_idx]
            hl_score = 100 if loss < no_dict_loss * 0.9 else 0
            row = [Cell(loss, hl_score), def_cell]
            rows.append(row)
            instance_idx += 1

        html.write_table(rows)


def generate_training_data(data_id):
    num_samples_list = open(os.path.join(working_path, "entry_prediction_n", data_id), "r").readlines()
    p = os.path.join(working_path, "entry_loss", "entry{}.pickle".format(data_id))
    loss_outputs_list = pickle.load(open(p, "rb"))
    print("Loaded input data")
    loss_outputs = []
    for e in loss_outputs_list:
        loss_outputs.extend(e["masked_lm_example_loss"])
    print("Total of {} loss outputs".format(len(loss_outputs)))
    feature_itr = load_record_v2(os.path.join(working_path, "entry_prediction_tf.done", data_id))

    instance_idx = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(working_path, "entry_prediction_train", data_id))

    n = len(num_samples_list)
    for i in range(n):
        n_sample = int(num_samples_list[i])
        assert n_sample > 0
        first_inst = feature_itr.__next__()

        if instance_idx + n_sample >= len(loss_outputs):
            break

        if n_sample == 1:
            continue

        no_dict_loss = loss_outputs[instance_idx]
        instance_idx += 1
        all_samples = []
        for j in range(1, n_sample):
            feature = feature_itr.__next__()
            loss = loss_outputs[instance_idx]
            if loss < no_dict_loss * 0.9:
                label = 1
            else:
                label = 0
            new_features = collections.OrderedDict()

            for key in feature:
                new_features[key] = btd.create_int_feature(take(feature[key]))

            new_features["useful_entry"] = btd.create_int_feature([label])

            example = tf.train.Example(features=tf.train.Features(feature=new_features))
            writer.write(example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    visualize_prediction_data("1")

