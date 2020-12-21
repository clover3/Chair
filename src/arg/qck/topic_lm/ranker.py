from collections import Counter
from typing import Dict, List

from arg.qck.filter_qk import LMScorer
from arg.qck.qcknc_datagen import QCKCandidateI
from misc_lib import TimeEstimator
from trec.trec_parse import TrecRankedListEntry


def rank_with_query_lm(query_lms: Dict[str, Counter],
         candidate_dict: Dict[str, List[QCKCandidateI]], num_query=100, alpha=0.5) -> Dict[str, List[TrecRankedListEntry]]:
    run_name = "run_name"
    scorer = LMScorer(query_lms, alpha)
    out_d = {}
    print("Start scoring")
    keys = list(candidate_dict.keys())
    keys = keys[:num_query]
    ticker = TimeEstimator(len(keys))
    for query_id in keys:
        candidates = candidate_dict[query_id]

        def get_score(c: QCKCandidateI) -> float:
            text = c.text
            assert text
            score = scorer.score_text(query_id, text)
            return score
        candidates.sort(key=get_score, reverse=True)
        l: List[TrecRankedListEntry] = []
        for rank, c in enumerate(candidates):
            l.append(TrecRankedListEntry(query_id, c.id, rank, get_score(c), run_name))
        out_d[query_id] = l
        ticker.tick()
    return out_d