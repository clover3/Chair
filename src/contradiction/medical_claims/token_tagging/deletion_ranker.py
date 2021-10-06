import json
import os
from typing import List, Tuple

from scipy.special import softmax

from contradiction.medical_claims.biobert.voca_common import get_biobert_tokenizer
from contradiction.medical_claims.token_tagging.token_tagging_common import get_split_score_to_pair_list_fn
from cpath import output_path
from data_generator.bert_input_splitter import get_sep_loc
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from explain.tf2.deletion_scorer import TokenExEntry, summarize_deletion_score
from list_lib import lmap
from misc_lib import group_by, get_first, get_second, get_third
from trec.trec_parse import scores_to_ranked_list_entries, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def get_neutral_prob(logit):
    probs = softmax(logit)
    return probs[1]


def get_cont_prob(logit):
    probs = softmax(logit)
    return probs[2]


def convert_split_input_ids_w_scores(tokenizer, input_ids, scores)\
        -> Tuple[List[str], List[str], List[int], List[int]]:
    tokens: List[str] = tokenizer.convert_ids_to_tokens(input_ids)
    idx_sep1, idx_sep2 = get_sep_loc(input_ids)
    sent1: List[str] = tokens[1:idx_sep1]
    scores1: List[int] = scores[1:idx_sep1]
    sent2: List[str] = tokens[idx_sep1+1: idx_sep2]
    scores2: List[int] = scores[idx_sep1+1: idx_sep2]
    return sent1, sent2, scores1, scores2


def check_is_valid_token_subword(token, subword_tokens):
    token_from_sbword = "".join(subword_tokens)
    token_from_sbword = token_from_sbword.replace("##", "")
    if token.lower() != token_from_sbword:
        print("Token is different from the subword: ", token.lower(), token_from_sbword)


def group_sbword_scores(t_text, scores, sent):
    # Lookup token idx
    e_list = []
    for idx, (sb_token, score) in enumerate(zip(sent, scores)):
        token_idx = t_text.sbword_mapping[idx]
        e = (token_idx, sb_token, score)
        e_list.append(e)

    # Group by token idx
    # Transpose array
    output = []
    for idx, entries in group_by(e_list, get_first).items():
        sbword_tokens: List[str] = lmap(get_second, entries)
        per_tokens_scores: List[float] = lmap(get_third, entries)
        e = idx, t_text.tokens[idx], sbword_tokens, per_tokens_scores
        output.append(e)
    return output


def make_ranked_list(dir_path, save_name, info_path, tag_type, tokenizer):
    run_name = "deletion"
    info_d = json.load(open(info_path, "r", encoding="utf-8"))
    deletion_per_job = 20
    num_jobs = 10
    save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name + ".txt")
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
    all_ranked_list: List[TrecRankedListEntry] = []
    split_score_to_pair_list = get_split_score_to_pair_list_fn(merge_method=sum)
    seen_data_id = set()

    for e in summarized_result:
        # assert e.data_id not in seen_data_id
        info = info_d[str(e.data_id)]
        seen_data_id.add(e.data_id)
        text1 = info['text1']
        text2 = info['text2']

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
        # print("prem")
        # print(t_text1.text)
        # print(sent1)
        # print(t_text1.tokens)
        # print(t_text1.sbword_tokens)
        # print(t_text1.sbword_mapping)
        # print("hypo")
        # print(t_text2.text)
        # print(sent2)
        # print(t_text2.tokens)
        # print(t_text2.sbword_tokens)
        # print(t_text2.sbword_mapping)
        # print()
        #

        def only_space_as_sep(text):
            return ("\t" not in text) and ("\n" not in text)

        if not only_space_as_sep(text1) or not only_space_as_sep(text2):
            raise ValueError

        todo = [
            (scores1, sent1, t_text1, 'prem'),
            (scores2, sent2, t_text2, 'hypo'),
        ]
        for scores, sent, t_text, sent_name in todo:
            doc_id_score_list: List[Tuple[str, float]] = split_score_to_pair_list(scores, sent, t_text)
            query_id = "{}_{}_{}_{}".format(info['group_no'], info['inner_idx'], sent_name, tag_type)
            ranked_list = scores_to_ranked_list_entries(doc_id_score_list, run_name, query_id)
            all_ranked_list.extend(ranked_list)

    write_trec_ranked_list_entry(all_ranked_list, save_path)


def main_for_bert():
    # tag_type = "mismatch"
    tag_type = "conflict"
    dir_path = os.path.join(output_path, "bert_alamri1_deletion_8")
    save_name = "deletion_" + tag_type
    info_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "bert_alamri1.info")
    tokenizer = get_tokenizer()
    make_ranked_list(dir_path, save_name, info_path, tag_type, tokenizer)


def main_for_biobert():
    tag_type = "mismatch"
    # tag_type = "conflict"
    tokenizer = get_biobert_tokenizer()

    dir_path = os.path.join(output_path, "biobert_alamri1_deletion")
    save_name = "biobert_deletion_" + tag_type
    info_path = os.path.join(output_path, "alamri_annotation1", "tfrecord", "biobert_alamri1.info")
    make_ranked_list(dir_path, save_name, info_path, tag_type, tokenizer)


if __name__ == "__main__":
    main_for_bert()