import os
import sys

from misc_lib import get_dir_files, exist_or_mkdir
from tf_util.enum_features import load_record, feature_to_ordered_dict
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature


def augment(short_records, long_records, target_len, save_dir, start_record_idx=0):
    exist_or_mkdir(save_dir)
    record_idx = start_record_idx
    print("record_idx", record_idx)

    def get_next_writer():
        return RecordWriterWrap(os.path.join(save_dir, str(record_idx)))

    writer = get_next_writer()
    cnt = 0
    while cnt < target_len:
        first_inst = short_records.__next__()
        second_inst = long_records.__next__()

        first_inst = feature_to_ordered_dict(first_inst)
        first_inst["next_sentence_labels"] = create_int_feature([1])
        second_inst = feature_to_ordered_dict(second_inst)
        second_inst["next_sentence_labels"] = create_int_feature([1])

        writer.write_feature(first_inst)
        writer.write_feature(second_inst)
  #
        cnt += 2
        if writer.total_written >= 100000:
            record_idx += 1
            print("Wrote {} data".format(cnt))
            writer.close()
            writer = get_next_writer()

    return



def enum_dir_records(dir_path):
    file_path_list = get_dir_files(dir_path)

    while True:
        for file_path in file_path_list:
            for item in load_record(file_path):
                yield item


def run(dir_path1, dir_path2, num_data, save_dir, start_record_idx=""):
    if start_record_idx:
        start_record_idx = int(start_record_idx)
    else:
        start_record_idx = 0

    augment(enum_dir_records(dir_path1), enum_dir_records(dir_path2), num_data, save_dir, start_record_idx)


if __name__ == '__main__':
    if len(sys.argv) == 6:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], sys.argv[5])
    else:
        run(sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4])

