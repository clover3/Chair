from typing import List, Tuple, Dict

from contradiction.medical_claims.token_tagging.query_id_helper import get_query_id
from contradiction.medical_claims.token_tagging.subtoken_helper import merge_subtoken_level_scores
from contradiction.medical_claims.token_tagging.trec_entry_helper import convert_token_score_d_to_trec_entries
from data_generator.tokenize_helper import TokenizedText
from trec.types import TrecRankedListEntry


def convert_inner(ex_logits_joined_d, info_d, run_name, tag_type, tokenizer):
    all_ranked_list: List[TrecRankedListEntry] = []
    ex_logits_joined: Tuple[str, ] = ex_logits_joined_d[tag_type]
    for data_id, ex_logits in ex_logits_joined:
        info = info_d[str(data_id[0])]
        text1 = info['text1']
        text2 = info['text2']
        t_text1 = TokenizedText.from_text(text1, tokenizer)
        t_text2 = TokenizedText.from_text(text2, tokenizer)
        sep_idx = len(t_text1.sbword_tokens) + 1
        sep_idx2 = sep_idx + len(t_text2.sbword_tokens) + 1
        scores1 = ex_logits[1:sep_idx]
        scores2 = ex_logits[sep_idx + 1:sep_idx2]

        todo = [
            (scores1, t_text1, 'prem'),
            (scores2, t_text2, 'hypo'),
        ]
        for scores, t_text, sent_name in todo:
            token_scores: Dict[int, float] = merge_subtoken_level_scores(max, scores, t_text)
            query_id = get_query_id(info['group_no'], info['inner_idx'],
                                            sent_name,
                                            tag_type)
            ranked_list = convert_token_score_d_to_trec_entries(query_id, run_name, token_scores)
            all_ranked_list.extend(ranked_list)
    return all_ranked_list