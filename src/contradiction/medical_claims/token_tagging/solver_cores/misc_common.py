from typing import Tuple, List

from scipy.special import softmax

from data_generator.bert_input_splitter import get_sep_loc
from explain.tf2.deletion_scorer import TokenExEntry
from list_lib import lmap
from misc_lib import group_by, get_first, get_second, get_third


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


def split_scores_for_two_sents(e: TokenExEntry, tokenizer):
    idx_max = max(e.contribution.keys())
    contribution_array = []
    for j in range(0, idx_max + 1):
        contribution_array.append(e.contribution[j])
    sent1, sent2, scores1, scores2 = convert_split_input_ids_w_scores(tokenizer,
                                                                      e.input_ids,
                                                                      contribution_array)
    return scores1, scores2, sent1, sent2