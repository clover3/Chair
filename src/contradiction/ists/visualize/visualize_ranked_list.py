from abc import ABC, abstractmethod
from typing import List, Dict, Callable

from alignment.base_ds import TextPairProblem
from contradiction.ists.save_path_helper import get_save_path, get_qrel_path
from contradiction.token_visualize import print_html, PerPairScores
from data_generator.tokenizer_wo_tf import get_tokenizer
from dataset_specific.ists.parse import iSTSProblem
from dataset_specific.ists.path_helper import load_ists_problems
from misc_lib import group_by
from trec.qrel_parse import load_qrels_structured
from trec.trec_parse import load_ranked_list
from trec.types import QRelsDict, TrecRankedListEntry


class QIDParse(ABC):
    @abstractmethod
    def tag_type(self):
        pass

    @abstractmethod
    def pair_no(self):
        pass

    @abstractmethod
    def sent_type(self):
        pass


class ISTSQIDParse(QIDParse):
    def __init__(self, qid):
        tag_type, sent_no, sent_type = qid.split("_")
        self._tag_type = tag_type
        self._pair_no = sent_no
        self._sent_type = sent_type

    def tag_type(self):
        return self._tag_type

    def pair_no(self):
        return self._pair_no

    def sent_type(self):
        return self._sent_type


def collect_scores(ranked_list: List[TrecRankedListEntry], qid_parser: Callable[[str], QIDParse],
                   doc_id_to_idx: Callable[[str], int]) -> Dict[str, PerPairScores]:
    grouped = group_by(ranked_list, lambda x: x.query_id)
    qid_to_score_d = {}
    for qid, entries in grouped.items():
        score_d = {}
        for e in entries:
            score_d[doc_id_to_idx(e.doc_id)] = e.score
        qid_to_score_d[qid] = score_d

    def get_pair_idx(qid):
        return qid_parser(qid).pair_no()

    pair_no_grouped = group_by(qid_to_score_d.keys(), get_pair_idx)
    output = {}
    for pair_no, qids in pair_no_grouped.items():
        per_pair_d = {}
        for qid in qids:
            qid_p = qid_parser(qid)
            per_pair_d[qid_p.tag_type(), qid_p.sent_type()] = qid_to_score_d[qid]
        output[pair_no] = PerPairScores(per_pair_d)
    return output


def get_qid(pair_no, sent_type, tag_type):
    return f"{tag_type}_{pair_no}_{sent_type}"


def doc_id_to_idx(doc_id):
    return int(doc_id) - 1


# tfrecord/bert_alamri1.pickle
def main():
    run_name = "nlits_punc_nc"
    tag_type = "noali"

    genre = "headlines"
    split = "train"
    tokenizer = get_tokenizer()
    ists_problems: List[iSTSProblem] = load_ists_problems(genre, split)
    problems: List[TextPairProblem] = [iSTSProblem.to_text_pair_problem(p) for p in ists_problems]

    qrel: QRelsDict = load_qrels_structured(get_qrel_path(genre, split))
    save_path = get_save_path(run_name)
    ranked_list = load_ranked_list(save_path)
    save_name = "{}_{}.html".format(run_name, tag_type)
    score_grouped: Dict[str, PerPairScores] = collect_scores(ranked_list, ISTSQIDParse, doc_id_to_idx)

    def loc_to_doc_id(loc):
        return str(loc+1)

    print_html(save_name, tag_type, score_grouped, problems, qrel, tokenizer,
               "1", "2", get_qid, loc_to_doc_id)


if __name__ == "__main__":
    main()
