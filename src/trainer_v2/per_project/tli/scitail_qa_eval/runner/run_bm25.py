from dataset_specific.scitail import get_scitail_questions
from trainer_v2.per_project.tli.qa_scorer.bm25_system import BM25TextPairScorer, BM25TextPairScorerClueWeb

from list_lib import right
from trainer_v2.per_project.tli.scitail_qa_eval.eval_helper import solve_save_scitail_qa


def not_tuned(split):
    system = BM25TextPairScorerClueWeb()
    solve_save_scitail_qa(system.score, split, "bm25_clue")


def task_tuned(split):
    system = BM25TextPairScorer(get_scitail_questions())
    solve_save_scitail_qa(system.score, split, "bm25_tuned")


def main():
    split = "test"
    not_tuned(split)
    task_tuned(split)


if __name__ == "__main__":
    main()