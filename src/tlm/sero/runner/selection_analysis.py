import argparse
import sys
from collections import defaultdict, Counter
from typing import List, Dict

import numpy as np
import scipy.special

from arg.qck.decl import get_format_handler
from arg.qck.prediction_reader import load_combine_info_jsons
from cache import save_to_pickle
from estimator_helper.output_reader import join_prediction_with_info
from misc_lib import group_by
from trec.qrel_parse import load_qrels_structured

parser = argparse.ArgumentParser(description='')


parser.add_argument("--pred_path")
parser.add_argument("--info_path")
parser.add_argument("--info_path2")
parser.add_argument("--save_name")
parser.add_argument("--input_type", default="qck")
parser.add_argument("--max_entry", default=100)
parser.add_argument("--combine_strategy", default="avg_then_doc_max")
parser.add_argument("--qrel_path", default="softmax")
parser.add_argument("--shuffle_sort", default=False)


def get_doc_length_info(info):
    d = defaultdict(list)
    for data_id, entry in info.items():
        key = entry['query'].query_id, entry['candidate'].id
        d[key].append(entry['idx'])

    out_d = {}
    for key, values in d.items():
        n_seg = len(values)
        n_seg_1 = max(values)
        assert n_seg == n_seg_1 + 1
        out_d[key] = n_seg
    return out_d


def main(pred_file_path: str,
         info_file_path: str,
         info_file_path2: str,
         save_name: str,
                        input_type: str,
                        qrel_path: str,
                        ):

    judgement = load_qrels_structured(qrel_path)
    def get_label(key):
        query_id, doc_id = key
        try:
            return judgement[query_id][doc_id]
        except KeyError:
            return 0

    f_handler = get_format_handler(input_type)
    info: Dict = load_combine_info_jsons(info_file_path, f_handler.get_mapping(), f_handler.drop_kdp())

    info2: Dict = load_combine_info_jsons(info_file_path2, f_handler.get_mapping(), f_handler.drop_kdp())
    doc_length = get_doc_length_info(info2)
    key_logit = "logits"

    data: List[Dict] = join_prediction_with_info(pred_file_path, info, ["data_id", key_logit])

    grouped = group_by(data, f_handler.get_pair_id)

    cnt = Counter()
    for key, entries in grouped.items():
        if not get_label(key):
            continue
        seg_groups = {}
        for e in entries:
            probs = scipy.special.softmax(e['logits'])[:, 1]
            seg_groups[e['idx']] = probs

        indices = list(seg_groups.keys())
        indices.sort()
        assert max(indices) == len(indices) - 1
        all_probs = []
        for seg_group_idx in seg_groups.keys():
            all_probs.extend(seg_groups[seg_group_idx])

        num_seg = doc_length[key]
        max_idx = np.argmax(all_probs[:num_seg])


        cnt[(max_idx, num_seg)] += 1

    save_to_pickle(cnt, save_name)





if __name__ == "__main__":
    args = parser.parse_args(sys.argv[1:])
    main(args.pred_path,
         args.info_path,
         args.info_path2,
         args.save_name,
        args.input_type,
        args.qrel_path,
        )










