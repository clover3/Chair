import json
import os
from typing import List, Dict, Tuple

import numpy as np

from contradiction.medical_claims.annotation_1.load_data import get_dev_group_no
from contradiction.medical_claims.biobert.voca_common import get_biobert_tokenizer
from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from contradiction.medical_claims.token_tagging.solver_cores.misc_common import get_neutral_prob, get_cont_prob, \
    convert_split_input_ids_w_scores
from contradiction.medical_claims.token_tagging.subtoken_helper import get_split_score_to_pair_list_fn
from cpath import output_path
from data_generator.tokenize_helper import TokenizedText
from explain.tf2.deletion_scorer import TokenExEntry, summarize_deletion_score
from list_lib import left
from misc_lib import NamedAverager, get_second
from trec.qrel_parse import load_qrels_structured
from trec.types import QRelsDict


def collect_nli_predictions(dir_path, info_path):

    tag_type = "mismatch"
    info_d, summarized_result = get_summarized_results(dir_path, info_path, tag_type)

    output_d: Dict[Tuple[int,int], List[float]] = {}
    for e in summarized_result:
        info_e = info_d[str(e.data_id)]
        data_no = info_e["group_no"], info_e["inner_idx"]
        output_d[data_no] = e.base_logits
    return output_d


def get_summarized_results(dir_path, info_path, tag_type):
    info_d = json.load(open(info_path, "r", encoding="utf-8"))
    deletion_per_job = 20
    num_jobs = 10
    max_offset = num_jobs * deletion_per_job
    batch_size = 8
    deletion_offset_list = list(range(0, max_offset, deletion_per_job))
    signal_function = {
        "conflict": get_cont_prob,
        "mismatch": get_neutral_prob
    }
    summarized_result: List[TokenExEntry] = \
        summarize_deletion_score(dir_path,
                                 deletion_per_job,
                                 batch_size,
                                 deletion_offset_list,
                                 signal_function[tag_type],
                                 )
    return info_d, summarized_result


TokenScoreDict = Dict[Tuple[int, int],
                      Tuple[List[str], List[str], List[Tuple[str, float]], List[Tuple[str, float]]]]


def collect_token_scores(dir_path, info_path, tokenizer, tag_type) -> TokenScoreDict:
    info_d, summarized_result = get_summarized_results(dir_path, info_path, tag_type)
    split_score_to_pair_list = get_split_score_to_pair_list_fn(merge_subtoken_scores=sum)
    output_d: TokenScoreDict = {}
    for e in summarized_result:
        info_e = info_d[str(e.data_id)]
        text1 = info_e['text1']
        text2 = info_e['text2']
        t_text1 = TokenizedText.from_text(text1, tokenizer)
        t_text2 = TokenizedText.from_text(text2, tokenizer)
        idx_min = min(e.contribution.keys())
        idx_max = max(e.contribution.keys())
        if idx_min != 0:
            print("WARNING idx_min != 0")
        if idx_max < len(t_text1.sbword_tokens) + len(t_text2.sbword_tokens):
            print("WARNING idx_max ({}) < {}".format(idx_max, len(t_text1.sbword_tokens) + len(t_text2.sbword_tokens)))

        contribution_array = []
        for j in range(0, idx_max+1):
            contribution_array.append(e.contribution[j])

        sent1, sent2, scores1, scores2 = convert_split_input_ids_w_scores(tokenizer,
                                                                          e.input_ids,
                                                                          contribution_array)
        data_no = info_e["group_no"], info_e["inner_idx"]
        doc_id_score_list1: List[Tuple[str, float]] = split_score_to_pair_list(scores1, sent1, t_text1)
        doc_id_score_list2: List[Tuple[str, float]] = split_score_to_pair_list(scores2, sent2, t_text2)
        output_d[data_no] = sent1, sent2, doc_id_score_list1, doc_id_score_list2
    return output_d


def AP(pred: List[str], gold: List[str]):
    n_pred_pos = 0
    tp = 0
    sum_prec = 0
    for idx in pred:
        n_pred_pos += 1
        if idx in gold:
            tp += 1
            sum_prec += (tp / n_pred_pos)
    return sum_prec / len(gold) if len(gold) else 1


def conditional_performance(nli_prediction_d: Dict[Tuple[int, int], List[float]],
                            conflict_score_d: TokenScoreDict,
                            mismatch_score_d: TokenScoreDict,
                            qrel,
                            ):
    data_no_list = list(nli_prediction_d.keys())
    valid_set_groups = get_dev_group_no()
    tag_targets = ["mismatch", "conflict"]

    data_no_list = [(group_no, inner_no)
                    for (group_no, inner_no) in data_no_list
                    if group_no in valid_set_groups]
    data_no_list.sort()
    score_d_d = {
        'conflict': conflict_score_d,
        'mismatch': mismatch_score_d
    }
    named_average = NamedAverager()
    for data_no in data_no_list:
        group_no, inner_idx = data_no
        nli_prediction = nli_prediction_d[data_no]
        pred = np.argmax(nli_prediction)

        # get average precision for mismatch/conflict.
        for tag in tag_targets:
            token_score_d = score_d_d[tag]
            sent1, sent2, scores1, scores2 = token_score_d[data_no]
            print(len(sent1), len(sent2), len(scores1), len(scores2), )
            for sent_type in ["prem", "hypo"]:
                qid = get_query_id(group_no, inner_idx, sent_type, tag)
                print(qid)
                if qid in qrel:
                    scores: List[Tuple[str, float]] = {
                        "prem": scores1,
                        "hypo": scores2,
                    }[sent_type]
                    gold = [k for k, v in qrel[qid].items() if v > 0]
                    scores.sort(key=get_second, reverse=True)
                    ranked_idx = left(scores)
                    ap = AP(ranked_idx, gold)
                    print(gold, ranked_idx, scores, ap)
                    named_average["{}_{}_{}".format(tag, sent_type, pred)].append(ap)
                    named_average["{}_{}".format(tag, sent_type)].append(ap)

    average_d = named_average.get_average_dict()
    keys = list(average_d)
    keys.sort()
    for key in keys:
        avg = average_d[key]
        n = len(named_average[key].history)
        print("{}\t{}\t{}".format(key, avg, n))



def main():
    # measure if the precision is better when the NLI prediction is correct.
    # I expect
    # If NLI prediction=contradiction, it will do good at 'conflict' prediction
    # If NLI prediction=neutral, it will do good at 'mismatch' prediction
    dir_path = os.path.join(output_path, "biobert_alamri1_deletion")
    info_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "biobert_alamri1.info")
    nli_prediction_d = collect_nli_predictions(dir_path, info_path)
    tokenizer = get_biobert_tokenizer()
    mismatch_score_d = collect_token_scores(dir_path, info_path, tokenizer, "mismatch")
    conflict_score_d = collect_token_scores(dir_path, info_path, tokenizer, "conflict")
    qrel_name = "worker_J.qrel"
    qrel_path = os.path.join(output_path, "alamri_annotation1", "label", qrel_name)
    qrel: QRelsDict = load_qrels_structured(qrel_path)
    conditional_performance(nli_prediction_d, conflict_score_d, mismatch_score_d, qrel)


if __name__ == "__main__":
    main()