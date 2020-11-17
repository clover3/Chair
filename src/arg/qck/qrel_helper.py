from typing import Iterable

from evals.trec import TrecRelevanceJudgementEntry


def get_trec_relevance_judgement(label_itr) -> Iterable[TrecRelevanceJudgementEntry]:
    for query_id, candidate_id, correctness in label_itr:
        if correctness:
            e = TrecRelevanceJudgementEntry(query_id, candidate_id, int(correctness))
            yield e