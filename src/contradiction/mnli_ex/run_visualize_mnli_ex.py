from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.token_tagging.path_helper import get_sbl_vak_qrel_path
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem
from contradiction.mnli_ex.load_mnli_ex_data import load_mnli_ex, MNLIExEntry
from contradiction.mnli_ex.ranking_style_helper import get_save_path
from contradiction.mnli_ex.trec_style_helper import get_mnli_ex_trec_style_label_path
from contradiction.token_visualize import TextPair, print_html
from data_generator.tokenizer_wo_tf import get_tokenizer
from misc_lib import group_by
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list
from trec.types import QRelsDict, TrecRankedListEntry


def to_text_pair(p: MNLIExEntry) -> TextPair:
    return TextPair(p.data_id, p.premise, p.hypothesis)


def collect_scores(ranked_list: List[TrecRankedListEntry], tag_type) -> Dict[str, Dict[Tuple, Dict]]:
    grouped = group_by(ranked_list, lambda x: x.query_id)
    qid_to_score_d = {}
    for qid, entries in grouped.items():
        score_d = {}
        for e in entries:
            score_d[e.doc_id] = e.score
        qid_to_score_d[qid] = score_d

    def get_pair_idx(qid):
        problem_id, sent_type = qid.split("_")
        return problem_id

    pair_no_grouped = group_by(qid_to_score_d.keys(), get_pair_idx)
    output = {}
    for pair_no, qids in pair_no_grouped.items():
        per_pair_d = {}
        for qid in qids:
            problem_id, sent_type = qid.split("_")
            per_pair_d[sent_type, tag_type] = qid_to_score_d[qid]
        output[pair_no] = per_pair_d
    return output


def debug_duplicate():
    run_name = "deletion"
    tag_type = "conflict"
    split = "dev"
    save_name = "{}_{}_{}".format(split, run_name, tag_type)
    ranked_list_path = get_save_path(save_name)
    ranked_list = load_ranked_list(ranked_list_path)
    grouped = group_by(ranked_list, lambda x: x.query_id)
    for query_id, items in grouped.items():
        doc_ids = set()
        for e in items:
            if e.doc_id in doc_ids:
                print(query_id)
                break
            doc_ids.add(e.doc_id)


def main():
    tokenizer = get_tokenizer()

    run_name = "senli"
    tag_type = "match"
    label = "c"
    split = "dev"
    save_name = "{}_{}_{}".format(split, run_name, tag_type)
    problems: List[MNLIExEntry] = load_mnli_ex(split, tag_type)
    text_pairs: List[TextPair] = list(map(to_text_pair, problems))

    qrel: QRelsDict = load_qrels_structured(get_mnli_ex_trec_style_label_path(label, split))

    ranked_list_path = get_save_path(save_name)
    ranked_list = load_ranked_list(ranked_list_path)
    score_grouped: Dict[str, Dict[Tuple[str, str], Dict]] = collect_scores(ranked_list, tag_type)

    save_name = "{}_{}_{}.html".format(split, run_name, tag_type)
    print_html(save_name, tag_type, score_grouped, text_pairs, qrel, tokenizer)


if __name__ == "__main__":
    # debug_duplicate()
    main()

