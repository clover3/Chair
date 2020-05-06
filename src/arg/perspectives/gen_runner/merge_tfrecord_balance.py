import collections
import os
import random
import sys

from list_lib import foreach
from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def main(dir_path):
    output_path = os.path.join(dir_path, "all_balanced")
    pos_insts = []
    neg_insts = []
    all_insts = [neg_insts, pos_insts]

    for i in range(665):
        p = os.path.join(dir_path, str(i))
        if os.path.exists(p):
            for record in load_record(p):
                new_features = collections.OrderedDict()
                for key in record:
                    new_features[key] = create_int_feature(take(record[key]))

                label = take(record['label_ids'])[0]
                all_insts[label].append(new_features)

    random.shuffle(pos_insts)
    random.shuffle(neg_insts)

    num_sel = min(len(pos_insts), len(neg_insts))
    print("{} insts per label".format(num_sel))

    insts_to_write = pos_insts[:num_sel] + neg_insts[:num_sel]
    writer = RecordWriterWrap(output_path)
    foreach(writer.write_feature, insts_to_write)


if __name__ == "__main__":
    main(sys.argv[1])
