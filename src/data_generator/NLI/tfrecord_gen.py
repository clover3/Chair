import os
import random

from cpath import output_path
from data_generator.NLI.nli import DataLoader
from data_generator.create_feature import create_int_feature
from data_generator.tfrecord_gen import entry_to_feature_dict, modify_data_loader, write_features_to_file
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import RecordWriterWrap


def get_modified_nli_data_loader(sequence_length):
    data_loader = DataLoader(sequence_length, "bert_voca.txt", True, True)
    return modify_data_loader(data_loader)


def gen_tf_recored():
    sequence_length = 200
    data_loader = get_modified_nli_data_loader(sequence_length)
    todo = [("train", [data_loader.train_file]), ("dev", [data_loader.dev_file])]
    batch_size = 32
    dir_path = os.path.join(output_path, "nli_tfrecord_{}".format(sequence_length))
    exist_or_mkdir(dir_path)

    for name, files in todo[::-1]:
        output_file = os.path.join(dir_path, name)
        writer = RecordWriterWrap(output_file)
        for file in files:
            for e in data_loader.example_generator(file):
                f = entry_to_feature_dict(e)
                f["is_real_example"] = create_int_feature([1])
                writer.write_feature(f)

        if name == "dev":
            while writer.total_written % batch_size != 0:
                f["is_real_example"] = create_int_feature([0])
                writer.write_feature(f)

        writer.close()

        print("Wrote %d total instances" % writer.total_written)


def split_train_to_tdev():
    sequence_length = 300
    data_loader = get_modified_nli_data_loader(sequence_length)
    file = data_loader.train_file

    dir_path = os.path.join(output_path, "nli_tfrecord_t_{}".format(sequence_length))
    exist_or_mkdir(dir_path)

    itr = data_loader.example_generator(file)
    all_inst = []
    for e in itr:
        f = entry_to_feature_dict(e)
        all_inst.append(f)

    random.shuffle(all_inst)

    tdev_size = 9000
    train_t = all_inst[:-tdev_size]
    dev_t = all_inst[-tdev_size:]
    assert len(train_t) + len(dev_t) == len(all_inst)

    def save(name, data):
        output_file = os.path.join(dir_path, name)
        writer = write_features_to_file(data, output_file)
        print("%s: Wrote %d total instances" % (name, writer.total_written))


    save("train_t", train_t)
    save("dev_t", dev_t)


def gen_tf_record_per_genre():
    sequence_length = 300
    data_loader = get_modified_nli_data_loader(sequence_length)
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
    gen_tf_recored()
