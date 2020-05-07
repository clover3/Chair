import os
import sys

from data_generator.tokenizer_wo_tf import pretty_tokens, get_tokenizer
from misc_lib import get_dir_files, exist_or_mkdir
from tf_util.enum_features import load_record, feature_to_ordered_dict
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.feature_to_text import take


def do_filtering(file_path, out_path, condition_fn, debug_call_back=None):
    writer = RecordWriterWrap(out_path)
    for item in load_record(file_path):
        features = feature_to_ordered_dict(item)
        if condition_fn(features):
            if debug_call_back is not None:
                debug_call_back(features)
            writer.write_feature(features)
    writer.close()


def run(in_dir_path, out_dir_path, keyword):
    exist_or_mkdir(out_dir_path)
    tokenizer = get_tokenizer()
    ids = tokenizer.convert_tokens_to_ids([keyword])
    assert len(ids) == 1
    id_keyword = ids[0]

    def condition_fn(features):
        return id_keyword in take(features['input_ids'])

    inst_cnt = 0

    def debug_call_back(features):
        nonlocal inst_cnt
        if inst_cnt < 4:
            input_tokens = tokenizer.convert_ids_to_tokens(take(features['input_ids']))
            print(pretty_tokens(input_tokens))
        inst_cnt += 1


    for file_path in get_dir_files(in_dir_path):
        inst_cnt = 0
        name = os.path.basename(file_path)
        out_path = os.path.join(out_dir_path, name)
        do_filtering(file_path, out_path, condition_fn)


if __name__ == '__main__':
    run(sys.argv[1], sys.argv[2], sys.argv[3])

