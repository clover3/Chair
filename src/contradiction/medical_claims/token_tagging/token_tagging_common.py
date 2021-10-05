from typing import List, Callable, Tuple, Iterable

from data_generator.tokenize_helper import TokenizedText
from list_lib import lmap
from misc_lib import group_by, get_first, get_second, get_third


def get_split_score_to_pair_list_fn(merge_method: Callable[[Iterable[float]], float]):
    # scores
    def split_score_to_pair_list(scores: List[float],
                                 sent: List[str],
                                 t_text: TokenizedText) -> List[Tuple[str, float]]:
        # Lookup token idx
        e_list: List[Tuple[int, str, float]] = []
        for idx, (sb_token, score) in enumerate(zip(sent, scores)):
            token_idx = t_text.sbword_mapping[idx]
            e = (token_idx, sb_token, score)
            e_list.append(e)

        # Group by token idx
        # Transpose array
        grouped = group_by(e_list, get_first)
        doc_id_score_list: List[Tuple[str, float]] = []
        for idx, entries in grouped.items():
            subword_tokens: List[str] = lmap(get_second, entries)
            per_tokens_scores: List[float] = lmap(get_third, entries)
            token = t_text.tokens[idx]
            # Check token ans subword_tokens begin equal
            token_from_sbword = "".join(subword_tokens)
            token_from_sbword = token_from_sbword.replace("##", "")
            if token.lower() != token_from_sbword:
                print("Token is different from the subword: ", token.lower(), token_from_sbword)
            doc_id_score_list.append((str(idx), merge_method(per_tokens_scores)))
        return doc_id_score_list

    return split_score_to_pair_list