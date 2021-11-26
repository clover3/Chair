import csv
import os
from collections import Counter, defaultdict
from typing import Dict

from cpath import output_path, at_data_dir
from data_generator.bert_input_splitter import get_first_seg
from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import job_man_dir
from tf_util.tfrecord_convertor import load_record_int64, extract_convertor
from tlm.qtype.is_functionword import FunctionWordClassifier

orig_inputs = [
    "input_ids1",
    "input_mask1",
    "segment_ids1",
    "input_ids2",
    "input_mask2",
    "segment_ids2",
    ]

drop_inputs = [
    "drop_input_ids1",
    "drop_input_mask1",
    "drop_segment_ids1",
    "drop_input_ids2",
    "drop_input_mask2",
    "drop_segment_ids2",
]


def convert_get_orig_inputs(features: Dict) -> Dict:
    out_d = {}
    for key in orig_inputs:
        out_d[key] = features[key]
    return out_d


def convert_get_drop_inputs(features: Dict) -> Dict:
    out_d = {}
    for src_key, out_key in zip(drop_inputs, orig_inputs):
        out_d[out_key] = features[src_key]
    return out_d


def build_query_d_info(input_tfrecord_path):
    query_cnt = Counter()
    tokenizer = get_tokenizer()
    query_d = {}

    def int_list_to_str(l):
        query_rep = " ".join(map(str, l))
        return query_rep

    def str_to_ids(key):
        return [int(s) for s in key.split()]

    def str_to_tokens(key):
        q_tokens_ids = str_to_ids(key)
        return tokenizer.convert_ids_to_tokens(q_tokens_ids)

    for record in load_record_int64(input_tfrecord_path):
        query_tokens = get_first_seg(record["input_ids1"])
        query_tokens_drop = get_first_seg(record["drop_input_ids1"])
        s = int_list_to_str(query_tokens) + "/" + int_list_to_str(query_tokens_drop)
        query_cnt[s] += 1

    func_count = Counter()
    for key, cnt in query_cnt.items():
        s_ori, s_drop = key.split("/")
        ori = str_to_tokens(s_ori)
        dropped = str_to_tokens(s_drop)
        functional_words = [t for t in ori if t not in dropped]
        func_count[" ".join(functional_words)] += cnt

        ori_ids = str_to_ids(s_ori)
        drop_ids = str_to_ids(s_drop)
        func_ids = [i for i in ori_ids if i not in drop_ids]
        query_d[s_ori] = (ori_ids, drop_ids, func_ids)

    fw_cls = FunctionWordClassifier()
    mapping = defaultdict(list)
    for key, value in func_count.items():
        tokens = key.split()
        top2 = sorted(tokens, key=lambda x: fw_cls.qdf[x])
        shorten = [t for t in tokens if t in top2[:2]]
        shorten_s = " ".join(shorten)
        mapping[shorten_s].append(key)
    for shorten_s, items in mapping.items():
        print()
        print(shorten_s)
        print([t for t in items])
    return query_d


def get_type_add_convertor():
    answer_path = at_data_dir("qtype", "1_type_annot.tsv")
    reader = csv.reader(open(answer_path, "r"), delimiter='\t')
    type_map_text = {}
    type_map = {}
    tokenizer = get_tokenizer()
    for row in reader:
        drop_token_str = row[1]
        type_str = row[2]
        type_map_text[drop_token_str] = type_str
        type_map[drop_token_str] = tokenizer.convert_tokens_to_ids(type_str.split())
        print(drop_token_str, type_str)

    n_iter = 0
    n_insert = 0
    def func(features: Dict) -> Dict:
        nonlocal n_iter
        n_iter = n_iter + 1
        seg1_ids = get_first_seg(features["drop_input_ids1"])
        query = " ".join(tokenizer.convert_ids_to_tokens(seg1_ids))
        if query in type_map:
            print(query)
            type_tokens = type_map[query]
        else:
            type_tokens = []

        l = len(type_tokens)
        def slice_insert_input_ids(input_ids):
            return input_ids[:1] + type_tokens + input_ids[1:-l]

        def slice_insert_input_mask(input_mask):
            return input_mask[:1] + [1] * l + input_mask[1:-l]

        def slice_insert_segment_ids(segment_ids):
            return segment_ids[:1] + [0] * l + segment_ids[1:-l]

        slide_fn_d = {
            'input_ids': slice_insert_input_ids,
            'input_mask': slice_insert_input_mask,
            'segment_ids': slice_insert_segment_ids,
        }
        out_d = convert_get_drop_inputs(features)

        nonlocal n_insert
        if type_tokens:
            n_insert += 1
            for f_name in slide_fn_d.keys():
                for i in [1, 2]:
                    key = "{}{}".format(f_name, i)
                    out_d[key] = slide_fn_d[f_name](out_d[key])
                    if n_insert < 10:
                        print(key, out_d[key])
        if n_iter % 1000 == 0:
            print((n_insert, n_iter))
        return out_d
    return func


def main():
    input_tfrecord_path = os.path.join(job_man_dir, "MMD_train_qtype1", "1")
    # query_d = build_query_d_info(input_tfrecord_path)
    type_add = get_type_add_convertor()
    todo = [
            ("type_add", type_add),
            ("original", convert_get_orig_inputs),
            ("drop", convert_get_drop_inputs),
            ]
    save_dir = os.path.join(output_path, "qtype_analysis_tfrecord")
    for name, convert_fn in todo:
        save_path = os.path.join(save_dir, name)
        extract_convertor(input_tfrecord_path, save_path, convert_fn)


if __name__ == "__main__":
    main()
