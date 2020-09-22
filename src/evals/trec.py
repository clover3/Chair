from typing import Iterable, NamedTuple


class TrecRelevanceJudgementEntry(NamedTuple):
    query_id: str
    doc_id: str
    relevance: int


def write_trec_relevance_judgement(entries: Iterable[TrecRelevanceJudgementEntry], save_path):
    f = open(save_path, "w")
    for e in entries:
        line = "{} 0 {} {}\n".format(e.query_id, e.doc_id, e.relevance)
        f.write(line)
    f.close()


class TrecRankedListEntry(NamedTuple):
    query_id: str
    doc_id: str
    rank: int
    score: float
    run_name: str


def write_trec_ranked_list_entry(entries: Iterable[TrecRankedListEntry], save_path):
    f = open(save_path, "w")
    for e in entries:
        line = "{} Q0 {} {} {} {}\n".format(e.query_id, e.doc_id, e.rank, e.score, e.run_name)
        f.write(line)
    f.close()