from typing import List, Dict

# tfrecord/bert_alamri1.pickle
from alignment.base_ds import TextPairProblem
from contradiction.ists.save_path_helper import get_not_entail_save_path, get_not_entail_qrel_path
from contradiction.ists.visualize.visualize_ranked_list import QIDParse, collect_scores, doc_id_to_idx
from contradiction.token_visualize import PerPairScores, print_html
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.ists.parse import iSTSProblem
from dataset_specific.ists.path_helper import load_ists_problems
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list
from trec.types import QRelsDict


class ISTSQIDParse(QIDParse):
    def __init__(self, qid):
        sent_no, sent_type = qid.split("-")
        self._pair_no = sent_no
        self._sent_type = sent_type

    def tag_type(self):
        return "not_entail"

    def pair_no(self):
        return self._pair_no

    def sent_type(self):
        return self._sent_type


def get_qid(pair_no, sent_type, _):
    return f"{pair_no}-{sent_type}"


def main():
    run_name = "nlits_punc"
    tag_type = "not_entail"

    genre = "headlines"
    split = "train"
    tokenizer = get_tokenizer()
    ists_problems: List[iSTSProblem] = load_ists_problems(genre, split)
    problems: List[TextPairProblem] = [iSTSProblem.to_text_pair_problem(p) for p in ists_problems]

    qrel: QRelsDict = load_qrels_structured(get_not_entail_qrel_path(genre, split))
    save_path = get_not_entail_save_path(genre, split, run_name)
    ranked_list = load_ranked_list(save_path)
    save_name = "{}_{}.html".format(run_name, tag_type)
    score_grouped: Dict[str, PerPairScores] = collect_scores(ranked_list, ISTSQIDParse, doc_id_to_idx)

    def loc_to_doc_id(loc):
        return str(loc+1)

    print_html(save_name, tag_type, score_grouped, problems, qrel, tokenizer,
               "1", "2", get_qid, loc_to_doc_id)


if __name__ == "__main__":
    main()
