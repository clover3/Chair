import collections
import os
import random

from data_generator.NLI.nli import DataLoader
from data_generator.common import get_tokenizer
from data_generator.create_feature import create_int_feature
from misc_lib import exist_or_mkdir
from path import output_path
from tf_util.record_writer_wrap import RecordWriterWrap


def entry_to_feature_dict(e):
    input_ids, input_mask, segment_ids, label = e
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["label_ids"] = create_int_feature([label])
    return features


def get_modified_data_loader(sequence_length):

    data_loader = DataLoader(sequence_length, "bert_voca.txt", True, True)
    tokenizer = get_tokenizer()
    CLS_ID = tokenizer.convert_tokens_to_ids(["[CLS]"])[0]
    SEP_ID = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    data_loader.CLS_ID = CLS_ID
    data_loader.SEP_ID = SEP_ID
    return data_loader


def gen_tf_recored():
    sequence_length = 300
    data_loader = get_modified_data_loader(sequence_length)
    todo = [("train", [data_loader.train_file]), ("dev", [data_loader.dev_file])]

    for name, files in todo[::-1]:
        dir_path = os.path.join(output_path, "nli_tfrecord_d2_{}".format(sequence_length))
        exist_or_mkdir(dir_path)
        output_file = os.path.join(dir_path, name)
        writer = RecordWriterWrap(output_file)
        for file in files:
            for e in data_loader.example_generator(file):
                f = entry_to_feature_dict(e)
                writer.write_feature(f)

        writer.close()

        print("Wrote %d total instances" % writer.total_written)


def split_train_to_tdev():
    sequence_length = 300
    data_loader = get_modified_data_loader(sequence_length)
    file = data_loader.train_file

    dir_path = os.path.join(output_path, "nli_tfrecord_t_{}".format(sequence_length))
    exist_or_mkdir(dir_path)
    all_inst = []
    for e in data_loader.example_generator(file):
        f = entry_to_feature_dict(e)
        all_inst.append(f)

    random.shuffle(all_inst)

    tdev_size = 9000
    train_t = all_inst[:-tdev_size]
    dev_t = all_inst[-tdev_size:]

    def save(name, data):
        output_file = os.path.join(dir_path, name)
        writer = RecordWriterWrap(output_file)
        for t in data:
            writer.write_feature(f)
        writer.close()
        print("%s: Wrote %d total instances" % (name, writer.total_written))
    save("train_t", train_t)
    save("dev_t", dev_t)


def gen_tf_record_per_genre():
    sequence_length = 300
    data_loader = get_modified_data_loader(sequence_length)
    todo = [("train", data_loader.train_file), ("dev", data_loader.dev_file)]

    for split_name, file in todo[::-1]:
        dir_path = os.path.join(output_path, "nli_tfrecord_per_genre".format(sequence_length))
        exist_or_mkdir(dir_path)
        genres = data_loader.get_genres(file)
        for genre in genres:
            file_name = "{}_{}".format(split_name, genre)
            output_file = os.path.join(dir_path, file_name)
            writer = RecordWriterWrap(output_file)
            for e in data_loader.example_generator_w_genre(file, genre):
                f = entry_to_feature_dict(e)
                writer.write_feature(f)

            writer.close()

            print("Wrote %d total instances" % writer.total_written)


if __name__ == "__main__":
    split_train_to_tdev()
