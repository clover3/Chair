from typing import Dict, List

from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from contradiction.medical_claims.token_tagging.solver_cores.misc_common import split_scores_for_two_sents, \
    get_cont_prob, \
    get_neutral_prob
from contradiction.medical_claims.token_tagging.subtoken_helper import merge_subtoken_level_scores
from contradiction.medical_claims.token_tagging.trec_entry_helper import convert_token_score_d_to_trec_entries
from data_generator.tokenize_helper import TokenizedText
from explain.tf2.deletion_scorer import TokenExEntry, summarize_deletion_score
from list_lib import foreach
from trec.types import TrecRankedListEntry


def convert_token_ex_entry_to_ranked_list(e: TokenExEntry,
                                          info,
                                          merge_fn, run_name, tag_type, tokenizer):
    idx_min = min(e.contribution.keys())
    idx_max = max(e.contribution.keys())
    check_sanity_index_min(idx_min)
    scores1, scores2, sent1, sent2 = split_scores_for_two_sents(e, tokenizer)
    text1: str = info['text1']
    text2: str = info['text2']
    t_text1: TokenizedText = TokenizedText.from_text(text1, tokenizer)
    t_text2: TokenizedText = TokenizedText.from_text(text2, tokenizer)
    len_sum = len(t_text1.sbword_tokens) + len(t_text2.sbword_tokens)
    check_sanity_index_max(idx_max, len_sum)

    def only_space_as_sep(text):
        return ("\t" not in text) and ("\n" not in text)

    if not only_space_as_sep(text1) or not only_space_as_sep(text2):
        raise ValueError
    todo = [
        (scores1, sent1, t_text1, 'prem'),
        (scores2, sent2, t_text2, 'hypo'),
    ]
    ranked_list_list = []
    for scores, sent, t_text, sent_name in todo:
        token_scores: Dict[int, float] = merge_subtoken_level_scores(merge_fn, scores, t_text)
        query_id = get_query_id(info['group_no'], info['inner_idx'], sent_name, tag_type)
        ranked_list = convert_token_score_d_to_trec_entries(query_id, run_name, token_scores)
        ranked_list_list.append(ranked_list)
    return ranked_list_list


def make_ranked_list_inner(dir_path, info_d, tag_type, tokenizer):
    run_name = "deletion"
    deletion_per_job = 20
    num_jobs = 10
    max_offset = num_jobs * deletion_per_job
    batch_size = 8
    deletion_offset_list = list(range(0, max_offset, deletion_per_job))
    signal_function = {
        "conflict": get_cont_prob,
        "mismatch": get_neutral_prob,
    }
    summarized_result: List[TokenExEntry] = \
        summarize_deletion_score(dir_path,
                                 deletion_per_job,
                                 batch_size,
                                 deletion_offset_list,
                                 signal_function[tag_type],
                                 )
    all_ranked_list = convert_token_ex_entry_list_to_ranked_list(info_d, run_name, summarized_result, tag_type,
                                                                 tokenizer)
    return all_ranked_list


def convert_token_ex_entry_list_to_ranked_list(info_d, run_name, summarized_result, tag_type, tokenizer):
    all_ranked_list: List[TrecRankedListEntry] = []
    merge_fn = sum
    for e in summarized_result:
        info = info_d[str(e.data_id)]
        ranked_list_list = convert_token_ex_entry_to_ranked_list(e, info, merge_fn, run_name, tag_type, tokenizer)
        foreach(all_ranked_list.extend, ranked_list_list)
    return all_ranked_list


def check_sanity_index_min(idx_min):
    if idx_min != 0:
        print("WARNING idx_min should be 0 but got {}".format(idx_min))


def check_sanity_index_max(idx_max, len_sum):
    if idx_max < len_sum:
        print("WARNING idx_max ({}) < {}".format(idx_max, len_sum))