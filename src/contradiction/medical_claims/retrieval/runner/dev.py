from contradiction.medical_claims.retrieval.bm25_system import BM25BioClaim, BM25Clueweb
from contradiction.medical_claims.retrieval.eval_helper import solve_bio_claim_and_save, build_qrel, \
    get_bioclaim_retrieval_corpus

"""
Task 1. Given an RQ, retrieve all relevant documents
     Method 1. Term-matching based (DONE)
     Method 2. Large-LM approach
     Method 3. NLI driven approach

Task 2. Given an RQ, build a contradictory pair
  How to check polarity?
     Append 'YES' or 'No' to the question and do NLI classification.
     Method 1. Large-LM approach
     Method 2. NLITS
"""

from contradiction.medical_claims.retrieval.path_helper import get_retrieval_qrel_path, get_retrieval_save_path
from list_lib import right
from runnable.trec.trec_eval_like import trec_eval_like_core
from trec.qrel_parse import load_qrels_flat_per_query
from trec.trec_parse import save_qrel, load_ranked_list_grouped


def do_save_qrel():
    for split in ["dev", "test"]:
        save_qrel(build_qrel(split), get_retrieval_qrel_path(split))


def main():
    system = BM25Clueweb()
    solve_bio_claim_and_save(system.score, "dev", "bm25_clue")


def analyze():
    rl = load_ranked_list_grouped(get_retrieval_save_path("bm25_2"))
    qrel = load_qrels_flat_per_query(get_retrieval_qrel_path("dev"))

    for k in [1, 5, 10, 20, 50, 100]:
        metric = "R{}".format(k)
        s = trec_eval_like_core(qrel, rl, metric)
        print(metric, s)


def task_tuned():
    split = "test"
    _, claims = get_bioclaim_retrieval_corpus(split)

    system = BM25BioClaim(right(claims))
    solve_bio_claim_and_save(system.score, split, "bm25_{}".format(split))


if __name__ == "__main__":
    task_tuned()
