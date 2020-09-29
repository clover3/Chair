import os
import pickle
from collections import defaultdict
from typing import List, Dict, Tuple

from arg.perspectives.doc_value_viewer.show_doc_value2 import collect_score_per_doc, fetch_score_per_pid, \
    get_score_from_entry, load_claim_d
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.get_doc_value import load_baseline
from arg.qck.prediction_reader import group_by_qid_cid, qck_convert_map, load_combine_info_jsons
from cpath import output_path
from data_generator.bert_input_splitter import split_p_h_with_input_ids
from data_generator.tokenizer_wo_tf import get_tokenizer
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap, foreach
from misc_lib import average, exist_or_mkdir
from tlm.token_utils import cells_from_tokens
from visualize.html_visual import HtmlVisualizer, Cell


def collect_passage_tokens(pid_entries)\
        -> Dict[Tuple[str, int], List[int]]:
    passage_tokens_dict = {}
    for pid, entries in pid_entries:
        for doc_idx, entry in enumerate(entries):
            input_ids = entry['input_ids2']
            key = entry['kdp'].doc_id, entry['kdp'].passage_idx
            try:
                _, passage_tokens = split_p_h_with_input_ids(input_ids, input_ids)

                if key in passage_tokens_dict:
                    a = passage_tokens_dict[key]
                    b = passage_tokens
                    if str(a[:200]) != str(b[:200]):
                        print(key)
                        print(str(a[:200]))
                        print(str(b[:200]))
                passage_tokens_dict[key] = passage_tokens
            except UnboundLocalError:
                print(input_ids)
    return passage_tokens_dict


def join_two_input_ids(input_ids1, input_ids2):
    min_gap = 1
    idx = len(input_ids1) - 1 - min_gap

    def is_overlap(idx):
        num_common_char = 0
        while idx + num_common_char < len(input_ids1):
            if input_ids1[idx + num_common_char] == input_ids2[num_common_char]:
                pass
            else:
                return False, num_common_char
            num_common_char += 1

        assert num_common_char < 100 * 1000
        return True, num_common_char

    while len(input_ids1) - idx <= len(input_ids2) and idx >= 0:
        f, num_common = is_overlap(idx)
        if f:
            global_overlap_start = idx
            local_overlap_ends = num_common
            return global_overlap_start, local_overlap_ends
        else:
            pass

        idx = idx - 1

    global_overlap_start = -1
    local_overlap_ends = -1
    return global_overlap_start, local_overlap_ends


def load_cppnc_score(fetch_field_list=None) -> Dict[str, Dict[str, List[Dict]]]:
    save_name = "qcknc_dense_val"

    score_name_list = []
    for i in range(0, 17):
        score_name_list.append("qcknc_dense_val_{}".format(i))

    out_dir = os.path.join(output_path, "cppnc")
    exist_or_mkdir(out_dir)
    info_file_path = os.path.join(out_dir, save_name + ".info")

    print("loading json info BEGIN")
    info = load_combine_info_jsons(info_file_path, qck_convert_map)
    print("loading json info DONE")
    all_predictions = []
    for score_name in score_name_list:
        pred_file_path = os.path.join(out_dir, score_name + ".score")
        print(score_name)
        predictions = join_prediction_with_info(pred_file_path, info, fetch_field_list)
        all_predictions.extend(predictions)

    qid_grouped = group_by_qid_cid(all_predictions)
    return qid_grouped

def load_cppnc_score_wrap():
    fetch_field_list = ['logits', 'input_ids2']
    obj = load_cppnc_score(fetch_field_list)
    cache_name = "cppnc_score_cache"
    # save_to_pickle(obj, cache_name)
    # obj = load_from_pickle(cache_name)
    return obj


def save_per_cid():
    print("Loading scores...")
    cid_grouped: Dict[str, Dict[str, List[Dict]]] = load_cppnc_score_wrap()
    save_root = os.path.join(output_path, "cppnc", "cid_grouped")
    exist_or_mkdir(save_root)

    for cid, entries in cid_grouped.items():
        save_path = os.path.join(save_root, cid)
        pickle.dump(entries, open(save_path, "wb"))


def main():
    print("Loading scores...")
    cid_grouped: Dict[str, Dict[str, List[Dict]]] = load_cppnc_score_wrap()
    baseline_cid_grouped = load_baseline("train_baseline")
    gold = get_claim_perspective_id_dict()
    tokenizer = get_tokenizer()
    claim_d = load_claim_d()

    print("Start analyzing")
    html = HtmlVisualizer("cppnc_value_per_token_score.html")
    claim_cnt = 0
    for cid, pid_entries_d in cid_grouped.items():
        pid_entries_d: Dict[str, List[Dict]] = pid_entries_d
        pid_entries: List[Tuple[str, List[Dict]]] = list(pid_entries_d.items())
        baseline_pid_entries = baseline_cid_grouped[int(cid)]
        baseline_score_d = fetch_score_per_pid(baseline_pid_entries)
        gold_pids = gold[int(cid)]

        ret = collect_score_per_doc(baseline_score_d, get_score_from_entry, gold_pids,
                                                                  pid_entries)
        passage_tokens_d = collect_passage_tokens(pid_entries)
        doc_info_d: Dict[int, Tuple[str, int]] = ret[0]
        doc_value_arr: List[List[float]] = ret[1]

        kdp_result_grouped = defaultdict(list)
        for doc_idx, doc_values in enumerate(doc_value_arr):
            doc_id, passage_idx = doc_info_d[doc_idx]
            avg_score = average(doc_values)
            kdp_result = doc_id, passage_idx, avg_score
            kdp_result_grouped[doc_id].append(kdp_result)

        s = "{} : {}".format(cid, claim_d[int(cid)])
        html.write_headline(s)
        claim_cnt += 1
        if claim_cnt > 10:
            break

        scores: List[float] = list([r[2] for r in doc_value_arr])

        foreach(html.write_paragraph, lmap(str, scores))

        for doc_id, kdp_result_list in kdp_result_grouped.items():
            html.write_headline(doc_id)
            tokens, per_token_score = combine_collect_score(tokenizer, doc_id, passage_tokens_d, kdp_result_list)
            str_tokens = tokenizer.convert_ids_to_tokens(tokens)
            row = cells_from_tokens(str_tokens)
            for idx in range(len(str_tokens)):
                score = per_token_score[idx][0]
                norm_score = min(abs(score) * 10000, 100)
                color = "B" if score > 0 else "R"
                row[idx].highlight_score = norm_score
                row[idx].target_color = color

            rows = [row]
            nth = 0
            any_score_found = True
            while any_score_found:
                any_score_found = False
                score_list = []
                for idx in range(len(str_tokens)):
                    if nth < len(per_token_score[idx]):
                        score = per_token_score[idx][nth]
                        any_score_found = True
                    else:
                        score = "-"
                    score_list.append(score)

                def get_cell(score):
                    if score == "-":
                        return Cell("-")
                    else:
                        # 0.01 -> 100
                        norm_score = min(abs(score) * 10000, 100)
                        color = "B" if score > 0 else "R"
                        return Cell("", highlight_score=norm_score, target_color=color)

                nth += 1
                if any_score_found:
                    row = lmap(get_cell, score_list)
                    rows.append(row)
            html.multirow_print_from_cells_list(rows)


def combine_collect_score(tokenizer, doc_id, passage_tokens_d, kdp_result_list):
    prev_passage_idx = -1
    prev_tokens = []
    per_token_scores = defaultdict(list)
    for kdp_result in kdp_result_list:
        doc_id_, passage_idx, avg_score = kdp_result
        input_ids_seg2 = passage_tokens_d[doc_id, passage_idx]


        # : compare with prev_tokens to get
        #  - overlap_start : idx in prev_tokens (gloabl index)
        #  - overlap_ends : idx in input_ids_seg2

        global_idx_overlap_start, local_idx_overlap_end = join_two_input_ids(prev_tokens, input_ids_seg2)
        if global_idx_overlap_start < 0 or local_idx_overlap_end < 0:
            if passage_idx == 0:
                pass
            else:
                print(tokenizer.convert_ids_to_tokens(prev_tokens))
                print(tokenizer.convert_ids_to_tokens(input_ids_seg2))
                print("Fail to join", doc_id_, passage_idx)
            global_idx_overlap_start = len(prev_tokens)
            local_idx_overlap_end = 0
        # global_idx_overlap_start = len(prev_tokens)
        # local_idx_overlap_end = 0

        new_tokens = input_ids_seg2[local_idx_overlap_end:]
        prev_tokens.extend(new_tokens)

        for g_idx in range(global_idx_overlap_start, len(prev_tokens)):
            per_token_scores[g_idx].append(avg_score)

        # if not prev_passage_idx + 1 == passage_idx:
        #     print(prev_passage_idx, passage_idx)
        #     assert False
        # prev_passage_idx = passage_idx

    return prev_tokens, per_token_scores


if __name__ == "__main__":
    save_per_cid()
