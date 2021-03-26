import os

from contradiction.medical_claims.biobert.mnli_data_loader import get_biobert_nli_data_loader
from cpath import output_path
from data_generator.create_feature import create_int_feature
from data_generator.tfrecord_gen import entry_to_feature_dict
from misc_lib import exist_or_mkdir
from tf_util.record_writer_wrap import RecordWriterWrap


def gen_tf_record():
    sequence_length = 300
    data_loader = get_biobert_nli_data_loader(sequence_length)
    todo = [("train", [data_loader.train_file]), ("dev", [data_loader.dev_file])]
    batch_size = 32
    dir_path = os.path.join(output_path, "biobert_mnli_{}".format(sequence_length))
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


def main():
    gen_tf_record()


if __name__ == "__main__":
    main()