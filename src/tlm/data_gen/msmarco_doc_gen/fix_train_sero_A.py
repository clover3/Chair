
import collections
import sys

from tf_util.enum_features import load_record_v2
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.bert_data_gen import create_int_feature
from tlm.data_gen.feature_to_text import take


def do_fix(source_path, output_path):
    max_num_seg = 4
    window_size = 512
    seq_length = 512 * max_num_seg
    input_names1 = [
                "input_ids1",
                   "segment_ids1",
                   "input_mask1",
                   ]
    input_names2 = [
                   "input_ids2",
                   "input_mask2",
                   "segment_ids2"
                   ]

    def feature_transformer(feature):
        new_features = collections.OrderedDict()
        def put(feature_name):
            return create_int_feature(take(feature[feature_name]))

        for left_right_idx in [1, 2]:
            input_names = [input_names1, input_names2][left_right_idx-1]
            input_ids = take(feature["input_ids{}".format(left_right_idx)])
            input_masks = take(feature["input_mask{}".format(left_right_idx)])
            cls_loc = []
            last_non_pad = -1
            for i in range(seq_length):
                if input_ids[i] == 101:
                    cls_loc.append(i)

                if input_masks[i] :
                    last_non_pad = i

            assert last_non_pad >= 0
            assert last_non_pad > cls_loc[-1]
            assert len(cls_loc) <= max_num_seg

            num_seg = len(cls_loc)
            input_building = {}
            for name in input_names:
                input_building[name] = []

            for i in range(num_seg):
                st = cls_loc[i]
                ed = cls_loc[i+1] if i+1 < num_seg else last_non_pad + 1
                pad_len = window_size - (ed - st)

                for input_name in input_names:
                    arr = take(feature[input_name])
                    seq = arr[st:ed] + pad_len * [0]
                    input_building[input_name].extend(seq)

            n_empty_seg = max_num_seg - num_seg
            for i in range(n_empty_seg):
                for input_name in input_names:
                    input_building[input_name].extend([0] * window_size)

            for input_name in input_names:
                checksum1 = sum(input_building[input_name])
                checksum2 = sum(take(feature[input_name]))
                assert checksum1 == checksum2

            for input_name in input_names:
                new_features[input_name] = create_int_feature(input_building[input_name])


        new_features["data_ids"] = put("data_ids")
        return new_features

    writer = RecordWriterWrap(output_path)
    feature_itr = load_record_v2(source_path)
    for feature in feature_itr:
        new_features_1 = feature_transformer(feature)
        writer.write_feature(new_features_1)
    writer.close()


def main():
    do_fix(sys.argv[1], sys.argv[2])
    return NotImplemented


if __name__ == "__main__":
    main()
