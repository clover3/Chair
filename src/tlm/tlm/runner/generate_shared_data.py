import collections
import os

from misc_lib import get_dir_files
from path import data_path, output_path
from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.dictionary.feature_to_text import take


def get_dir_all_itr(dir_path):
    for file_path in get_dir_files(dir_path):
        one_itr = load_record_v2(file_path)
        for item in one_itr:
            yield item


def encode(lm_data_path, nli_data_path):
    itr_lm = get_dir_all_itr(lm_data_path)
    itr_nli = load_record_v2(nli_data_path)
    out_path = os.path.join(output_path, "lm_nli")
    writer = RecordWriterWrap(out_path)

    for nli_entry in itr_nli:
        lm_entry = itr_lm.__next__()

        new_features = collections.OrderedDict()
        for key in lm_entry:
            new_features[key] = create_int_feature(take(lm_entry[key]))

        for key in nli_entry:
            if key == "label_ids":
                new_features[key] = create_int_feature(take(nli_entry[key]))
            else:
                new_key = "nli_" + key
                new_features[new_key] = create_int_feature(take(nli_entry[key]))

        writer.write_feature(new_features)

    print("Wrote {} items".format(writer.total_written))
    writer.close()


if __name__ == '__main__':
    lm_dir = os.path.join(data_path, "unmasked_pair_x3")
    nli_path = os.path.join(output_path, "nli_tfrecord_512", "train")
    encode(lm_dir, nli_path)

