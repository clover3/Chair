from data_generator.NLI.nli import DataLoader
import tensorflow as tf
from path import output_path
import os
import collections
from data_generator.create_feature import create_int_feature
from misc_lib import exist_or_mkdir

def gen_tf_recored():
    data_loader = DataLoader(200, "bert_voca.txt", True)
    todo = [("train", data_loader.train_file), ("dev", data_loader.dev_file)]

    for name, file in todo[::-1]:
        exist_or_mkdir(os.path.join(output_path, "nli_tfrecord_200"))
        output_file = os.path.join(output_path, "nli_tfrecord_200", name)
        writer = tf.io.TFRecordWriter(output_file)
        total_written = 0
        for e in data_loader.example_generator(file):
            input_ids, input_mask, segment_ids, label = e

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(input_ids)
            features["input_mask"] = create_int_feature(input_mask)
            features["segment_ids"] = create_int_feature(segment_ids)
            features["label_ids"] = create_int_feature([label])

            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())
            total_written += 1

        writer.close()

        print("Wrote %d total instances" % total_written)

if __name__ == "__main__":
    gen_tf_recored()