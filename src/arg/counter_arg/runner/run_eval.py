from typing import List, Callable

from arg.counter_arg.eval import run_eval, EvalCondition
from arg.counter_arg.header import Passage
from arg.counter_arg.methods import bm25_predictor, msmarco_predictor
from arg.counter_arg.methods.structured import get_structured_scorer
from arg.perspectives.collection_based_classifier import NamedNumber


def bm25_eval():
    split = "training"
    scorer: Callable[[Passage, List[Passage]], List[float]] = bm25_predictor.get_scorer(split)
    # print(run_eval(split, scorer, EvalCondition.SameThemeCounters))
    # print(run_eval(split, scorer, EvalCondition.SameThemeArguments))
    print(run_eval(split, scorer, EvalCondition.EntirePortalCounters))
    #print(run_eval(split, scorer, EvalCondition.EntirePortalArguments))


def bm25_structured_eval():
    split = "training"
    k = 1
    query_reweight = 0.5
    scorer: Callable[[Passage, List[Passage]], List[NamedNumber]] = get_structured_scorer(split, query_reweight, k)
    acc = run_eval(split, scorer, EvalCondition.EntirePortalCounters)
    print(query_reweight, acc)


def msmarco_eval():
    split = "training"
    scorer: Callable[[Passage, List[Passage]], List[NamedNumber]] = msmarco_predictor.get_scorer()
    # print(run_eval(split, scorer, EvalCondition.SameThemeCounters))
    #print(run_eval(split, scorer, EvalCondition.SameThemeArguments))
    print(run_eval(split, scorer, EvalCondition.EntirePortalCounters))
    #print(run_eval(split, scorer, EvalCondition.EntirePortalArguments))


if __name__ == "__main__":
    bm25_eval()
