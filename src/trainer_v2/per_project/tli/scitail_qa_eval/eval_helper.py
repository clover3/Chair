from typing import List

from dataset_specific.scitail import ScitailEntry, load_scitail_structured
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import TextPairScorer, BatchTextPairScorer, \
    get_batch_text_scorer
from trainer_v2.per_project.tli.scitail_qa_eval.path_helper import get_score_save_path


def load_scitail_qa_label(split):
    entries: List[ScitailEntry] = load_scitail_structured(split)
    output = []
    for e in entries:
        output.append(e.get_relevance_label())
    return output


def batch_solve_scitail_qa(scorer: BatchTextPairScorer, split) -> List[float]:
    entries: List[ScitailEntry] = load_scitail_structured(split)
    batch_todo = []
    for e in entries:
        batch_todo.append((e.question, e.sentence1))

    return scorer(batch_todo)


def solve_save_scitail_qa(scorer: TextPairScorer, run_name, split=""):
    batch_scorer = get_batch_text_scorer(scorer)
    if split:
        _batch_solve_save_scitail_qa(batch_scorer, run_name, split)
    else:
        for split in ["dev", "test"]:
            _batch_solve_save_scitail_qa(batch_scorer, run_name, split)


def batch_solve_save_scitail_qa(scorer: BatchTextPairScorer, run_name, split=""):
    if split:
        _batch_solve_save_scitail_qa(scorer, run_name, split)
    else:
        for split in ["dev", "test"]:
            _batch_solve_save_scitail_qa(scorer, run_name, split)


def _batch_solve_save_scitail_qa(scorer: BatchTextPairScorer, run_name, split):
    scores = batch_solve_scitail_qa(scorer, split)
    save_name = f"{run_name}_{split}"
    save_path = get_score_save_path(save_name)
    write_scores(save_path, scores)


def write_scores(save_path, scores):
    f = open(save_path, "w")
    for s in scores:
        f.write(f"{s}\n")
    f.close()


def load_scores(save_path):
    f = open(save_path, "r")
    output = []
    for line in f:
        output.append(float(line))
    return output
