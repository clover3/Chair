import csv
import os

from cache import load_from_pickle
from cpath import output_path
from crs.contradiction_pair.datagen_common import save_to_tfrecord
from list_lib import lmap


def load_label():
    label_path = "C:\work\Data\cont annotation\\Batch_3960955_batch_results.csv"
    reader = csv.reader(open(label_path, "r", encoding="utf-8"))

    rows = list(reader)
    column = rows[0]
    label_idx = column.index("Answer.sentiment.label")
    hit_id_idx = column.index("HITId")
    print(label_idx)
    print(hit_id_idx)
    print(len(rows))
    text_to_label = {
        'Contradiction':2,
        'No contradiction':0
    }
    for i in range(100):
        first = rows[1+ i*2]
        second = rows[2 + i * 2]
        assert first[hit_id_idx] == second[hit_id_idx]
        first_label = text_to_label[first[label_idx]]
        second_label = text_to_label[second[label_idx]]
        yield first_label, second_label


def merge_or(first_label, second_label):
    if first_label == 2 or second_label == 2:
        label = 2
    else:
        label = 0
    return label


def merge_and(first_label, second_label):
    if first_label == 2 and second_label == 2:
        label = 2
    else:
        label = 0
    return label


def write_to_tfrecord_all():
    input_ids_list = load_from_pickle("cont_annot_input_ids")
    labels = get_label_as_or()

    def format_dict(pair):
        return {'input_ids':pair[0],
                'label': pair[1]}

    test_data = lmap(format_dict, zip(input_ids_list, labels))
    save_path = os.path.join(output_path, "cont_pair_eval")
    save_to_tfrecord(test_data, save_path)


def get_label_as_or():
    raw_labels = list(load_label())
    labels = []
    for first_label, second_label in raw_labels:
        labels.append(merge_or(first_label, second_label))
    return labels


def write_to_tfrecord_unanimous_only():
    input_ids_list = load_from_pickle("cont_annot_input_ids")
    raw_labels = list(load_label())
    data = []
    for (first_label, second_label), input_id in zip(raw_labels, input_ids_list):
        if first_label == second_label:
            data.append((input_id, first_label))


    def format_dict(pair):
        return {'input_ids':pair[0],
                'label': pair[1]}

    test_data = lmap(format_dict, data)
    save_path = os.path.join(output_path, "cont_pair_eval_unanimous")
    save_to_tfrecord(test_data, save_path)



if __name__ == "__main__":
    write_to_tfrecord_unanimous_only()