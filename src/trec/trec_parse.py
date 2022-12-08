from typing import Iterator
from typing import List, Iterable, Dict, Tuple

from list_lib import flatten
from misc_lib import group_by, get_second, dict_to_tuple_list
from trec.types import TrecRankedListEntry, TrecRelevanceJudgementEntry


def write_trec_relevance_judgement(entries: Iterable[TrecRelevanceJudgementEntry], save_path):
    f = open(save_path, "w")
    for e in entries:
        line = "{} 0 {} {}\n".format(e.query_id, e.doc_id, e.relevance)
        f.write(line)
    f.close()

save_qrel = write_trec_relevance_judgement

def write_trec_ranked_list_entry(entries: Iterable[TrecRankedListEntry], save_path: str):
    f = open(save_path, "w")
    for e in entries:
        line = "{} Q0 {} {} {} {}\n".format(e.query_id, e.doc_id, e.rank, e.score, e.run_name)
        f.write(line)
    f.close()


def parse_ranked_list(line_itr: Iterator[str]) -> List[TrecRankedListEntry]:
    output: List[TrecRankedListEntry] = []
    for line in line_itr:
        q_id, _, doc_id, rank, score, run_name = line.split()
        e = TrecRankedListEntry(query_id=q_id, doc_id=str(doc_id), rank=int(rank), score=float(score),
                                run_name=run_name)
        output.append(e)
    return output


def load_ranked_list(path) -> List[TrecRankedListEntry]:
    ranked_list: List[TrecRankedListEntry] = parse_ranked_list(open(path, "r"))
    return ranked_list


RLG = Dict[str, List[TrecRankedListEntry]]
def load_ranked_list_grouped(path) -> Dict[str, List[TrecRankedListEntry]]:
    ranked_list: List[TrecRankedListEntry] = load_ranked_list(path)

    def get_qid(e: TrecRankedListEntry):
        return e.query_id

    return group_by(ranked_list, get_qid)


def write_rlg(rlg: RLG, save_path: str):
    return write_trec_ranked_list_entry(flatten(rlg.values()), save_path)


def trec_writer(doc_id, doc_text) -> List[str]:
    lines = []
    lines.append("<DOC>\n")
    lines.append("<DOCNO>{}</DOCNO>\n".format(doc_id))
    lines.append("<TEXT>\n")
    lines.append(doc_text)
    lines.append("</TEXT>\n")
    lines.append("</DOC>\n")
    return lines


def scores_to_ranked_list_entries(scores: List[Tuple[str, float]], run_name, query_id) -> List[TrecRankedListEntry]:
    scores.sort(key=get_second, reverse=True)

    output = []
    for rank, (doc_id, score) in enumerate(scores):
        e = TrecRankedListEntry(query_id, doc_id, rank, score, run_name)
        output.append(e)
    return output


def score_d_to_ranked_list_entries(scores_d: Dict[str, float], run_name, query_id) -> List[TrecRankedListEntry]:
    scores = dict_to_tuple_list(scores_d)
    scores.sort(key=get_second, reverse=True)

    output = []
    for rank, (doc_id, score) in enumerate(scores):
        e = TrecRankedListEntry(query_id, doc_id, rank, score, run_name)
        output.append(e)
    return output