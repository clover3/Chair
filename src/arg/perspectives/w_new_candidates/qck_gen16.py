import os
from collections import Counter
from typing import List, Dict

from krovetzstemmer import Stemmer

from arg.bm25 import BM25
from arg.clueweb12_B13_termstat import load_clueweb12_B13_termstat_stemmed, cdf
from arg.perspectives.load import d_n_claims_per_split2
from arg.perspectives.load import load_claim_ids_for_split
from arg.perspectives.qck.qck_common import get_qck_candidate_from_ranked_list_path
from arg.perspectives.qck.qcknc_datagen import is_correct_factory
from arg.perspectives.query.write_rm_extended_query import extract_terms_from_structured_query
from arg.qck.decl import QKUnit
from arg.qck.instance_generator.qcknc_w_rel_score import QCKInstGenWScore
from arg.qck.qck_worker import QCKWorker
from cache import load_from_pickle
from cpath import output_path
from epath import job_man_dir
from evals.types import TrecRankedListEntry
from galagos.parse import load_query_json_as_dict
from job_manager.job_runner_with_server import JobRunnerS
from list_lib import lmap
from misc_lib import two_digit_float
from trec.trec_parse import TrecRankedListEntry
from trec.trec_parse import load_ranked_list_grouped


def normalize_scores(query_path, kdp_ranked_list: Dict[str, List[TrecRankedListEntry]]) \
        -> Dict[str, List[TrecRankedListEntry]]:
    print(query_path)
    json_query = load_query_json_as_dict(query_path)
    max_value_d = get_max_values(json_query)


    out_d = {}
    for qid, rl in kdp_ranked_list.items():
        divider: float = max_value_d[qid]

        assert divider > 0
        largest_score = rl[0].score
        if largest_score > divider:
            divider = largest_score * 4
        bad_cnt = 0
        scores = []
        def get_new_entry(e: TrecRankedListEntry):
            new_score = e.score / divider
            scores.append(new_score)
            return TrecRankedListEntry(e.query_id, e.doc_id, e.rank, new_score, e.run_name)

        out_d[qid] = lmap(get_new_entry, rl)

        print(qid, bad_cnt, len(rl))
        print(" ".join(lmap(two_digit_float, scores)))

    return out_d


def get_max_values(queries: Dict[str, str]) -> Dict[str, float]:
    tf, df = load_clueweb12_B13_termstat_stemmed()
    stemmer = Stemmer()
    avdl = 500
    bm25_module = BM25(df, cdf, avdl)
    score_d = {}
    for qid, query_text in queries.items():
        q_terms = extract_terms_from_structured_query(query_text)
        q_terms_stemmed: List[str] = lmap(stemmer.stem, q_terms)
        q_tf = Counter(q_terms_stemmed)
        d_tf = q_tf
        score = bm25_module.score_inner(q_tf, d_tf)
        score_d[qid] = score
    return score_d


def qck_gen(job_name, qk_candidate_name, query_path,
            candidate_ranked_list_path, kdp_ranked_list_path, split):
    claim_ids = load_claim_ids_for_split(split)
    cids: List[str] = lmap(str, claim_ids)
    qk_candidate: List[QKUnit] = load_from_pickle(qk_candidate_name)
    raw_kdp_ranked_list: Dict[str, List[TrecRankedListEntry]] = load_ranked_list_grouped(kdp_ranked_list_path)
    kdp_ranked_list = normalize_scores(query_path, raw_kdp_ranked_list)

    print("cids", len(cids))
    print("len(qk_candidate)", len(qk_candidate))
    print("Generate instances : ", split)
    generator = QCKInstGenWScore(get_qck_candidate_from_ranked_list_path(candidate_ranked_list_path),
                                 is_correct_factory(),
                                 kdp_ranked_list
                                 )
    qk_candidate_train: List[QKUnit] = list([qk for qk in qk_candidate if qk[0].query_id in cids])

    def worker_factory(out_dir):
        return QCKWorker(qk_candidate_train,
                         generator,
                         out_dir)

    num_jobs = d_n_claims_per_split2[split]
    runner = JobRunnerS(job_man_dir, num_jobs, job_name + "_" + split, worker_factory)
    runner.start()


def main():
    split = "train"
    job_name = "qck16"
    qk_candidate_name = "pc_qk2_filtered_" + split
    query_path = os.path.join(output_path, "perspective_experiments",
                              "claim_query", "perspective_claim_query2_{}.json".format(split))
    candidate_ranked_list_path = os.path.join(output_path,
                                    "perspective_experiments",
                                    "pc_qres", "{}.txt".format(split))

    kdp_ranked_list_path = os.path.join(output_path,
                                        "perspective_experiments",
                                        "clueweb_qres", "{}.txt".format(split))
    qck_gen(job_name, qk_candidate_name, query_path,
            candidate_ranked_list_path, kdp_ranked_list_path, split)


if __name__ == "__main__":
    main()