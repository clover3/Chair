from collections import Counter
from typing import Iterable, List, Dict, Tuple

from adhoc.bm25_class import BM25
from dataset_specific.msmarco.passage.load_term_stats import load_msmarco_passage_term_stat
from dataset_specific.msmarco.passage.passage_resource_loader import FourItem
from list_lib import left
from misc_lib import get_second, TELI
from trainer_v2.per_project.transparency.mmp.term_effect_measure.compute_gains import EffectiveRankedListInfo


def get_rel_pid(qrels, qid):
    qrel = qrels[qid]
    for pid, score in qrel.items():
        if score > 0:
            return pid
    return None


def filter_rl_by_effective_scale(entries, rel_score, effective_scale):
    effective_max = rel_score + effective_scale
    effective_min = rel_score - effective_scale
    doc_ids = []
    scores = []
    base_rank = 0
    for pid, score in entries:
        if score > effective_max:
            base_rank += 1
        elif score >= effective_min:
            doc_ids.append(pid)
            scores.append(score)
        else:
            break
    return base_rank, doc_ids, scores


def precompute_ranked_list(itr: Iterable[List[FourItem]], bm25: BM25, target_q_term, qrels):
    e_ranked_list_list: List[EffectiveRankedListInfo] = []
    doc_id_to_tf: Dict[str, Counter] = {}
    qid_doc_id_to_target_tf: Dict[Tuple[str, str], int] = {}

    effective_scale = bm25.term_idf_factor(target_q_term) * 2
    for group in TELI(itr, 1000):
        first_entry = group[0]
        entries = []
        for qid, pid, query, text in group:
            q_terms = bm25.tokenizer.tokenize_stem(query)
            t_terms = bm25.tokenizer.tokenize_stem(text)
            q_tf = Counter(q_terms)
            t_tf = Counter(t_terms)
            doc_id_to_tf[pid] = t_tf
            qid_doc_id_to_target_tf[qid, pid] = t_tf[target_q_term]
            score = bm25.score_inner(q_tf, t_tf)
            entries.append((pid, score))

        entries.sort(key=get_second, reverse=True)

        qid = first_entry[0]
        rel_pid = get_rel_pid(qrels, qid)
        if rel_pid is None:
            continue

        try:
            rel_idx = left(entries).index(rel_pid)
        except ValueError:
            continue

        rel_score = entries[rel_idx][1]
        base_rank, doc_ids, scores = filter_rl_by_effective_scale(entries, rel_score, effective_scale)

        e_ranked_list = EffectiveRankedListInfo(
            qid,
            doc_ids,
            scores,
            base_rank,
            rel_idx - base_rank
        )
        e_ranked_list_list.append(e_ranked_list)
    return e_ranked_list_list, doc_id_to_tf, qid_doc_id_to_target_tf