import collections

from misc_lib import pick1
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.base import get_basic_input_feature_as_list
from tlm.data_gen.bert_data_gen import create_int_feature


def write_pairwise_record(tokenizer, max_seq_length, insts, out_path):
    writer = RecordWriterWrap(out_path)
    for inst in insts:
        (tokens, segment_ids), (tokens2, segment_ids2) = inst

        input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                             tokens, segment_ids)
        features = collections.OrderedDict()
        features["input_ids1"] = create_int_feature(input_ids)
        features["input_mask1"] = create_int_feature(input_mask)
        features["segment_ids1"] = create_int_feature(segment_ids)
        input_ids, input_mask, segment_ids = get_basic_input_feature_as_list(tokenizer, max_seq_length,
                                                                             tokens2, segment_ids2)
        features["input_ids2"] = create_int_feature(input_ids)
        features["input_mask2"] = create_int_feature(input_mask)
        features["segment_ids2"] = create_int_feature(segment_ids)

        writer.write_feature(features)
    writer.close()


def generate_pairwise_combinations(neg_inst_list, pos_inst_list):
    insts = []
    print("pos_insts", len(pos_inst_list))
    print("neg_insts", len(neg_inst_list))
    if len(pos_inst_list) > len(neg_inst_list):
        major_inst = pos_inst_list
        minor_inst = neg_inst_list
        pos_idx = 0
    else:
        major_inst = neg_inst_list
        minor_inst = pos_inst_list
        pos_idx = 1
    for idx, entry in enumerate(major_inst):
        entry2 = pick1(minor_inst)

        pos_entry = [entry, entry2][pos_idx]
        neg_entry = [entry, entry2][1 - pos_idx]
        insts.append((pos_entry, neg_entry))
    return insts