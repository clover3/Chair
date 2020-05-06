import collections
import os
import sys

from tf_util.enum_features import load_record
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def main(dir_path):
    output_path = os.path.join(dir_path, "all")
    writer = RecordWriterWrap(output_path)
    for i in range(665):
        p = os.path.join(dir_path, str(i))
        if os.path.exists(p):
            for record in load_record(p):
                new_features = collections.OrderedDict()
                for key in record:
                    new_features[key] = create_int_feature(take(record[key]))
                writer.write_feature(new_features)


if __name__ == "__main__":
    main(sys.argv[1])
