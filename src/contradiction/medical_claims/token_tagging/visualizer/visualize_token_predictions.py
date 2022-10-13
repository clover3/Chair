import os
from typing import List, Dict, Tuple

from alignment.base_ds import TextPairProblem
from alignment.data_structure.related_eval_instance import TextPair
from contradiction.medical_claims.token_tagging.path_helper import get_sbl_qrel_path
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, load_alamri_problem
from contradiction.token_visualize import print_html, PerPairScores
from cpath import output_path
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import group_by
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list
from trec.types import QRelsDict, TrecRankedListEntry


def collect_scores(ranked_list: List[TrecRankedListEntry]) -> Dict[str, PerPairScores]:
    grouped = group_by(ranked_list, lambda x: x.query_id)
    qid_to_score_d = {}
    for qid, entries in grouped.items():
        score_d = {}
        for e in entries:
            score_d[int(e.doc_id)] = e.score
        qid_to_score_d[qid] = score_d

    def get_pair_idx(qid):
        group_no, inner_idx, sent_type, tag_type = qid.split("_")
        group_no = int(group_no)
        inner_idx = int(inner_idx)
        return "{}_{}".format(group_no, inner_idx)

    pair_no_grouped = group_by(qid_to_score_d.keys(), get_pair_idx)
    output = {}
    for pair_no, qids in pair_no_grouped.items():
        per_pair_d: Dict[Tuple[str,str], Dict[int, float]] = {}
        for qid in qids:
            group_no, inner_idx, sent_type, tag_type = qid.split("_")
            score_d: Dict[int, float] = qid_to_score_d[qid]
            per_pair_d[tag_type, sent_type] = score_d
        output[pair_no] = PerPairScores(per_pair_d)
    return output


def alamri_to_text_pair(p: AlamriProblem) -> TextPairProblem:
    return TextPairProblem(p.get_problem_id(), p.text1, p.text2)


# tfrecord/bert_alamri1.pickle
def main():
    run_name = "nlits95"
    tag_type = "mismatch"


    tokenizer = get_tokenizer()
    problems: List[AlamriProblem] = load_alamri_problem()
    text_pairs: List[TextPairProblem] = list(map(alamri_to_text_pair, problems))
    ranked_list_path = os.path.join(output_path, "alamri_annotation1", "ranked_list",
                                    "{}_{}.txt".format(run_name, tag_type))
    qrel: QRelsDict = load_qrels_structured(get_sbl_qrel_path())
    ranked_list = load_ranked_list(ranked_list_path)
    save_name = "{}_{}.html".format(run_name, tag_type)
    score_grouped: Dict[str, PerPairScores] = collect_scores(ranked_list)
    print_html(save_name, tag_type, score_grouped, text_pairs, qrel, tokenizer)


if __name__ == "__main__":
    main()

