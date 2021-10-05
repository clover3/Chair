import json
import os
from typing import List, Dict, Tuple

from cache import load_from_pickle
from contradiction.medical_claims.token_tagging.token_tagging_common import get_split_score_to_pair_list_fn
from cpath import output_path
from data_generator.NLI import nli_info
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer
from trec.trec_parse import scores_to_ranked_list_entries, write_trec_ranked_list_entry
from trec.types import TrecRankedListEntry


def convert(pred_list: List[List[Tuple]],
            info_d: Dict[str, Dict],
            tokenizer):
    run_name = "senli"
    split_score_to_pair_list = get_split_score_to_pair_list_fn(max)

    tag_types = ["mismatch", "conflict"]
    ex_logits_joined_d = dict(zip(nli_info.tags, pred_list))

    for tag_type in tag_types:
        save_name = "senli_{}".format(tag_type)
        all_ranked_list: List[TrecRankedListEntry] = []
        ex_logits_joined = ex_logits_joined_d[tag_type]
        for data_id, ex_logits in ex_logits_joined:
            info = info_d[str(data_id[0])]
            text1 = info['text1']
            text2 = info['text2']
            t_text1 = TokenizedText.from_text(text1, tokenizer)
            t_text2 = TokenizedText.from_text(text2, tokenizer)
            sep_idx = len(t_text1.sbword_tokens) + 1
            sep_idx2 = sep_idx + len(t_text2.sbword_tokens) + 1
            scores1 = ex_logits[1:sep_idx]
            scores2 = ex_logits[sep_idx+1:sep_idx2]


            todo = [
                (scores1, t_text1, 'prem'),
                (scores2, t_text2, 'hypo'),
            ]
            for scores, t_text, sent_name in todo:
                doc_id_score_list: List[Tuple[str, float]]\
                    = split_score_to_pair_list(scores, t_text.sbword_tokens, t_text)
                query_id = "{}_{}_{}_{}".format(info['group_no'], info['inner_idx'],
                                                sent_name,
                                                tag_type)
                ranked_list = scores_to_ranked_list_entries(doc_id_score_list, run_name, query_id)
                all_ranked_list.extend(ranked_list)
        save_path = os.path.join(output_path, "alamri_annotation1", "ranked_list", save_name + ".txt")
        write_trec_ranked_list_entry(all_ranked_list, save_path)


# tfrecord/bert_alamri1.pickle
def main():
    data_name = "bert_alamri1"
    tokenizer = get_tokenizer()
    save_dir = os.path.join(output_path, "alamri_annotation1", "tfrecord")
    info_file_path = os.path.join(save_dir, "{}.info".format(data_name))
    run_name = "alamri_bert_nli"
    pred_list = load_from_pickle(run_name)
    info = json.load(open(info_file_path, "r"))
    convert(pred_list, info, tokenizer)


if __name__ == "__main__":
    main()
