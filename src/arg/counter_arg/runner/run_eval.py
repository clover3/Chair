from typing import List, Callable

from arg.counter_arg.eval import run_eval, EvalCondition
from arg.counter_arg.header import Passage
from arg.counter_arg.methods import bm25_predictor


def bm25_eval():
    split = "training"
    scorer: Callable[[Passage, List[Passage]], float] = bm25_predictor.get_scorer(split)

    # print(run_eval(split, scorer, EvalCondition.SameThemeCounters))
    print(run_eval(split, scorer, EvalCondition.SameThemeArguments))
    #print(run_eval(split, scorer, EvalCondition.EntirePortalCounters))
    #print(run_eval(split, scorer, EvalCondition.EntirePortalArguments))


if __name__ == "__main__":
    bm25_eval()
