from trainer_v2.per_project.tli.qa_scorer.bm25_system import BM25TextPairScorer, BM25TextPairScorerClueWeb
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import solve_bio_claim_and_save, build_qrel, \
    get_bioclaim_retrieval_corpus

from list_lib import right


def not_tuned(split):
    system = BM25TextPairScorerClueWeb()
    solve_bio_claim_and_save(system.score, split, "bm25_clue")


def task_tuned(split):
    _, claims = get_bioclaim_retrieval_corpus(split)
    system = BM25TextPairScorer(right(claims))
    solve_bio_claim_and_save(system.score, split, "bm25_tuned")


def main():
    split = "test"
    not_tuned(split)
    task_tuned(split)


if __name__ == "__main__":
    main()