import collections
import os
from typing import List, Iterable, Dict, Tuple

from adhoc.bm25 import BM25_2
from cache import load_from_pickle
from data_generator.subword_convertor import SubwordConvertor
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import group_by, find_max_idx, tprint, DataIDManager, TimeEstimator
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.base import pad0
from tlm.data_gen.classification_common import InstAsInputIds, encode_inst_as_input_ids
from tlm.data_gen.robust_gen.select_supervision.read_score import decompress_seg_ids_entry, save_info
from tlm.robust.load import robust_query_intervals


def enum_best_segments(get_score_fn, info, max_seg=4) -> Iterable[Dict]:
    grouped = group_by(info.values(), lambda e: (e['query_id'], e['doc_id']))
    for qid, doc_id in grouped:
        key = qid, doc_id
        sub_entries = grouped[key]

        def get_score(e):
            return get_score_fn(e['input_ids'])

        sub_entries = [e for e in sub_entries if e['passage_idx'] < max_seg]
        max_idx = find_max_idx(get_score, sub_entries)

        selected_raw_entry = sub_entries[max_idx]
        yield selected_raw_entry


def get_score_fn_functor():
    df_d = load_from_pickle("subword_df_robust_train")
    df_d = collections.Counter(df_d)
    tokenizer = get_tokenizer()
    sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    sbc = SubwordConvertor()
    collection_size = 1139368311 + 10
    avdl = 446
    def get_score(input_ids):
        sep_idx1 = input_ids.index(sep_id)
        sep_idx2 = input_ids.index(sep_id, sep_idx1 + 1)

        query = input_ids[1:sep_idx1]
        doc_content = input_ids[sep_idx1+1:sep_idx2]

        q_terms: List[Tuple[int]] = list(sbc.get_word_as_subtoken_tuple(query))
        d_terms: List[Tuple[int]] = list(sbc.get_word_as_subtoken_tuple(doc_content))

        tf = collections.Counter()
        for d_term in d_terms:
            if d_term in q_terms:
                tf[d_term] += 1

        score = 0
        for q_term in q_terms:
            f = tf[q_term]
            df = df_d[q_term]
            N = collection_size
            dl = len(d_terms)
            score += BM25_2(f, df, N, dl, avdl)

        return score

    return get_score


def generate_selected_training_data(info, max_seq_length, save_dir, get_score_fn, max_seg):
    data_id_manager = DataIDManager(0, 1000000)
    tprint("data gen")

    def get_query_id_group(query_id):
        for st, ed in robust_query_intervals:
            if st <= int(query_id) <= ed:
                return st
        assert False

    maybe_num_insts = int(len(info) / 4)
    ticker = TimeEstimator(maybe_num_insts)
    itr = enum_best_segments(get_score_fn, info, max_seg)
    insts = collections.defaultdict(list)
    for selected_entry in itr:
        ticker.tick()
        selected = decompress_seg_ids_entry(selected_entry)
        query_id = selected['query_id']
        q_group = get_query_id_group(query_id)
        assert len(selected['input_ids']) == len(selected['seg_ids'])

        selected['input_ids'] = pad0(selected['input_ids'], max_seq_length)
        selected['seg_ids'] = pad0(selected['seg_ids'], max_seq_length)
        # data_id = data_id_manager.assign(selected_segment.to_info_d())
        data_id = 0
        ci = InstAsInputIds(
            selected['input_ids'],
            selected['seg_ids'],
            selected['label'],
            data_id)
        insts[q_group].append(ci)

    def encode_fn(inst: InstAsInputIds) -> collections.OrderedDict:
        return encode_inst_as_input_ids(max_seq_length, inst)

    tprint("writing")
    for q_group, insts_per_group in insts.items():
        out_path = os.path.join(save_dir, str(q_group))
        write_records_w_encode_fn(out_path, encode_fn, insts_per_group, len(insts_per_group))
        save_info(save_dir, data_id_manager, str(q_group) + ".info")


def generate_selected_training_data_w_json(info, max_seq_length, save_dir, get_score_fn, max_seg):
    data_id_manager = DataIDManager(0, 1000000)
    tprint("data gen")

    def get_query_id_group(query_id):
        for st, ed in robust_query_intervals:
            if st <= int(query_id) <= ed:
                return st

        assert False

    tokenizer = get_tokenizer()
    for data_id, e in info.items():
        input_ids = tokenizer.convert_tokens_to_ids(e['tokens'])
        e['input_ids'] = input_ids

    maybe_num_insts = int(len(info) / 4)
    ticker = TimeEstimator(maybe_num_insts)
    itr = enum_best_segments(get_score_fn, info, max_seg)
    insts = collections.defaultdict(list)
    for selected_entry in itr:
        ticker.tick()
        selected = selected_entry
        query_id = selected['query_id']
        q_group = get_query_id_group(query_id)
        assert len(selected['tokens']) == len(selected['seg_ids'])
        input_ids = tokenizer.convert_tokens_to_ids(selected['tokens'])
        selected['input_ids'] = pad0(input_ids, max_seq_length)
        selected['seg_ids'] = pad0(selected['seg_ids'], max_seq_length)
        # data_id = data_id_manager.assign(selected_segment.to_info_d())
        data_id = 0
        ci = InstAsInputIds(
            selected['input_ids'],
            selected['seg_ids'],
            selected['label'],
            data_id)
        insts[q_group].append(ci)

    def encode_fn(inst: InstAsInputIds) -> collections.OrderedDict:
        return encode_inst_as_input_ids(max_seq_length, inst)

    tprint("writing")
    for q_group, insts_per_group in insts.items():
        out_path = os.path.join(save_dir, str(q_group))
        write_records_w_encode_fn(out_path, encode_fn, insts_per_group, len(insts_per_group))
        save_info(save_dir, data_id_manager, str(q_group) + ".info")


def demo_score(info, max_seq_length):
    tprint("data gen")
    df_d = load_from_pickle("subword_df_robust_train")
    df_d = collections.Counter(df_d)
    tokenizer = get_tokenizer()
    sep_id = tokenizer.convert_tokens_to_ids(["[SEP]"])[0]
    sbc = SubwordConvertor()
    collection_size = 1139368311 + 10
    avdl = 446
    def get_score(input_ids):
        sep_idx1 = input_ids.index(sep_id)
        sep_idx2 = input_ids.index(sep_id, sep_idx1 + 1)

        query = input_ids[1:sep_idx1]
        doc_content = input_ids[sep_idx1+1:sep_idx2]

        q_terms: List[Tuple[int]] = list(sbc.get_word_as_subtoken_tuple(query))
        d_terms: List[Tuple[int]] = list(sbc.get_word_as_subtoken_tuple(doc_content))

        tf = collections.Counter()
        for d_term in d_terms:
            if d_term in q_terms:
                tf[d_term] += 1

        score = 0
        for q_term in q_terms:
            f = tf[q_term]
            df = df_d[q_term]
            N = collection_size
            dl = len(d_terms)
            score += BM25_2(f, df, N, dl, avdl)

        def to_str(input_ids):
            return " ".join(tokenizer.convert_ids_to_tokens(input_ids))

        print("query", to_str(query))
        print("doc", to_str(doc_content))
        print('score:', score)

    insts = collections.defaultdict(list)
    grouped = group_by(info.values(), lambda e: (e['query_id'], e['doc_id']))
    for qid, doc_id in grouped:
        key = qid, doc_id
        sub_entries = grouped[key]
        for e in sub_entries:
            get_score(e['input_ids'])

    def encode_fn(inst: InstAsInputIds) -> collections.OrderedDict:
        return encode_inst_as_input_ids(max_seq_length, inst)
