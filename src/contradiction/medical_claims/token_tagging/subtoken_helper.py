from collections import defaultdict
from typing import List, Callable, Tuple, Iterable, Dict

from data_generator.tokenize_helper import TokenizedText
from list_lib import lmap, dict_value_map
from misc_lib import group_by, get_first, get_second, get_third


def assert_token_subword_same(idx, entries, t_text):
    subword_tokens: List[str] = lmap(get_second, entries)
    token = t_text.tokens[idx]
    # Check token ans subword_tokens begin equal
    token_from_sbword = "".join(subword_tokens)
    token_from_sbword = token_from_sbword.replace("##", "")
    if token.lower() != token_from_sbword:
        print("Token is different from the subword: ", token.lower(), token_from_sbword)


# Convert subtoken-level score into word level score, output is represented as tuple of
def get_split_score_to_pair_list_fn(merge_subtoken_scores: Callable[[Iterable[float]], float]) -> \
        Callable[[List[float], List[str], TokenizedText], List[Tuple[str, float]]]:
    # scores
    def split_score_to_pair_list(scores: List[float],
                                 sbtoken_list: List[str],
                                 t_text: TokenizedText) -> List[Tuple[str, float]]:
        # Lookup token idx
        e_list: List[Tuple[int, str, float]] = []
        for idx, (sb_token, score) in enumerate(zip(sbtoken_list, scores)):
            token_idx = t_text.sbword_mapping[idx]
            e = (token_idx, sb_token, score)
            e_list.append(e)

        # Group by token idx
        # Transpose array
        grouped = group_by(e_list, get_first)
        doc_id_score_list: List[Tuple[str, float]] = []
        for idx, entries in grouped.items():
            assert_token_subword_same(idx, entries, t_text)
            per_tokens_scores: List[float] = lmap(get_third, entries)
            s: float = merge_subtoken_scores(per_tokens_scores)
            doc_id_score: Tuple[str, float] = str(idx), s
            doc_id_score_list.append(doc_id_score)
        return doc_id_score_list

    return split_score_to_pair_list


# Convert subtoken-level score into word level score, output is represented as tuple of
InputArgs = List[float], List[str], TokenizedText
def merge_subtoken_level_scores(merge_subtoken_scores: Callable[[Iterable[float]], float],
                                scores: List[float],
                                t_text: TokenizedText) -> Dict[int, float]:
    grouped: Dict[int, List[float]] = defaultdict(list)
    for idx, (sb_token, score) in enumerate(zip(t_text.sbword_tokens, scores)):
        grouped[idx].append(score)

    return dict_value_map(merge_subtoken_scores, grouped)
