import pickle
from path import output_path, data_path
import os
from misc_lib import pick1
import tensorflow as tf
from data_generator import tokenizer_wo_tf
from tf_util.enum_features import load_record
from visualize.html_visual import Cell, HtmlVisualizer
import collections
import random
from tlm.wiki import bert_training_data as btd


def take(v):
    return v.int64_list.value


def generate_training_data(data_id):
    num_samples_list = open(os.path.join(output_path, "lookup_n", data_id), "r").readlines()
    p = os.path.join(output_path, "example_loss{}.pickle".format(data_id))
    loss_outputs_list = pickle.load(open(p, "rb"))
    loss_outputs = []
    for e in loss_outputs_list:
        loss_outputs.extend( e["masked_lm_example_loss"])
    print("Total of {} loss outputs".format(len(loss_outputs)))
    feature_itr = load_record(os.path.join(output_path, "lookup_example", data_id))

    instance_idx = 0
    writer = tf.python_io.TFRecordWriter(os.path.join(output_path, "lookup_train", data_id))

    n = len(num_samples_list)
    for i in range(n):
        f_feed_dictionary = random.random() < 0.5
        n_sample = int(num_samples_list[i])
        assert n_sample > 0
        first_inst = feature_itr.__next__()
        max_seq_len = len(take(first_inst["input_ids"]))

        if instance_idx + n_sample >= len(loss_outputs):
            break

        if n_sample == 1:
            continue

        no_dict_loss = loss_outputs[instance_idx]
        instance_idx += 1
        all_samples = []
        good_locations = []
        for j in range(1, n_sample):
            feature = feature_itr.__next__()
            d_location_ids = take(feature["d_location_ids"])
            loss = loss_outputs[instance_idx]
            if loss < no_dict_loss * 0.9:
                good_locations.extend([idx for idx in d_location_ids if idx > 0])
            all_samples.append(feature)
            instance_idx += 1

        lookup_idx = list([0 for _ in range(max_seq_len)])
        for loc in good_locations:
            lookup_idx[loc] = 1

        if f_feed_dictionary:
            base_feature = pick1(all_samples)
        else:
            base_feature = first_inst

        new_features = collections.OrderedDict()
        for key in base_feature:
            new_features[key] = btd.create_int_feature(take(base_feature[key]))

        new_features["lookup_idx"] = btd.create_int_feature(lookup_idx)

        example = tf.train.Example(features=tf.train.Features(feature=new_features))
        writer.write(example.SerializeToString())

    writer.close()



def load_and_visualize():
    tokenizer = tokenizer_wo_tf.FullTokenizer(os.path.join(data_path, "bert_voca.txt"))

    data_id = "1"

    n_list = open(os.path.join(output_path, "lookup_n", data_id), "r").readlines()
    p = os.path.join(output_path, "example_loss.pickle")
    data = pickle.load(open(p, "rb"))
    data = data[0]["masked_lm_example_loss"]

    feature_itr = load_record(os.path.join(output_path, "lookup_example", data_id))

    n = len(n_list)
    feature_idx = 0
    html_writer = HtmlVisualizer("lookup_loss2.html", dark_mode=False)

    for i in range(n):
        n_sample = int(n_list[i])
        rows = []
        assert n_sample > 0
        for j in range(n_sample):
            feature = feature_itr.__next__()

            input_ids = take(feature["input_ids"])
            masked_lm_ids = take(feature["masked_lm_ids"])
            masked_lm_positions = take(feature["masked_lm_positions"])
            input_mask = take(feature["input_mask"])
            selected_word = take(feature["selected_word"])
            d_input_ids = take(feature["d_input_ids"])
            d_location_ids = take(feature["d_location_ids"])

            word_tokens = tokenizer.convert_ids_to_tokens(selected_word)
            word = tokenizer_wo_tf.pretty_tokens((word_tokens))

            emph_word = "<b>" + word + "</b>"

            if j ==0 :
                mask_ans = {}
                masked_terms = tokenizer.convert_ids_to_tokens(masked_lm_ids)
                for pos, id in zip(list(masked_lm_positions), masked_terms):
                    mask_ans[pos] = id

                tokens = tokenizer.convert_ids_to_tokens(input_ids)

            for i in range(len(tokens)):
                if tokens[i] == "[MASK]":
                    tokens[i] = "[MASK_{}: {}]".format(i, mask_ans[i])
                if i in d_location_ids and i is not 0:
                    if tokens[i - 1] != emph_word:
                        tokens[i] = emph_word
                    else:
                        tokens[i] = "-"

            def_str = tokenizer_wo_tf.pretty_tokens(tokenizer.convert_ids_to_tokens(d_input_ids), True)
            row = list()
            row.append(Cell(word))
            row.append(Cell(data[feature_idx]))
            row.append(Cell(def_str))
            rows.append(row)

            feature_idx += 1

        s = tokenizer_wo_tf.pretty_tokens(tokens, True)
        html_writer.write_paragraph(s)

        html_writer.write_table(rows)

    html_writer.close()


if __name__ == '__main__':
    generate_training_data("2")

