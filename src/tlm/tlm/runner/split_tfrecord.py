import sys

from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.tfrecord_viewer import read_bert_data, repack_features


def split(file_path):
    a_out = file_path + "_a"
    b_out = file_path + "_b"
    writer_a = RecordWriterWrap(a_out)
    writer_b = RecordWriterWrap(b_out)
    tfrecord_itr = read_bert_data(file_path)
    for idx, inst in enumerate(tfrecord_itr):
        inst = repack_features(inst)
        if idx % 2 == 0:
            writer_a.write_feature(inst)
        else:
            writer_b.write_feature(inst)

    writer_a.close()
    writer_b.close()


if __name__ == "__main__":
    split(sys.argv[1])