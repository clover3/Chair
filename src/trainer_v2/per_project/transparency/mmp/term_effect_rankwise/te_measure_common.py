from typing import Dict, List

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_list_to_gz_jsonl
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.fidelity_helper import TermEffectPerQuery
from trainer_v2.per_project.transparency.mmp.term_effect_rankwise.term_effect_measure import IndexedRankedList, \
    ScoringModelIF


def term_effect_per_partition(
        partition_no, q_term_index_per_job,
        sm: ScoringModelIF,
        q_term, d_term,
        irl_proxy, get_te_save_path_fn):
    save_path = get_te_save_path_fn(q_term, d_term, partition_no)
    affected_qid_list: List[str] = q_term_index_per_job[q_term]
    te_list: List[TermEffectPerQuery] = []
    for qid in affected_qid_list:
        ranked_list: IndexedRankedList = irl_proxy.get_irl(qid)
        old_scores: List[float] = ranked_list.get_shallow_model_base_scores()
        entry_indices = ranked_list.get_entries_w_term(d_term)
        changes = []
        for entry_idx in entry_indices:
            entry = ranked_list.entries[entry_idx]
            new_score: float = sm.get_updated_score_bm25(q_term, d_term, entry)
            changes.append((entry_idx, new_score))

        target_scores = ranked_list.get_deep_model_scores()
        per_query = TermEffectPerQuery(target_scores, old_scores, changes)
        te_list.append(per_query)
    out_itr = map(TermEffectPerQuery.to_json, te_list)
    c_log.debug("Save to %s", save_path)
    save_list_to_gz_jsonl(out_itr, save_path)
    return te_list
