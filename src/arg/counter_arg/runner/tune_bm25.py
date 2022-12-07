from adhoc.bm25_class import BM25
from arg.counter_arg.eval import run_eval, EvalCondition
from arg.counter_arg.methods import bm25_predictor
from arg.counter_arg.methods.bm25_predictor import get_bm25_module


def modify(bm25_module: BM25, k1, k2, b):
    bm25_module.k1 = k1
    bm25_module.k2 = k2
    bm25_module.b = b


def run(k1, k2, b):
    split = "training"
    bm25 = get_bm25_module(split)
    modify(bm25, k1, k2, b)
    scorer = bm25_predictor.get_scorer_from_bm25_module(bm25)
    acc = run_eval(split, scorer, EvalCondition.EntirePortalArguments)
    print(f"k1 {k1}, k2 {k2}, b {b}, acc {acc}")


def grid_search_b():
    k1 = 1.2
    k2 = 100
    for b in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
        run(k1, k2, b)


def grid_search_k1():
    k1 = 0.5
    k2 = 100
    b = 0.7

    while k1 <= 1.1:
        run(k1, k2, b)
        k1 += 0.1


def grid_search_k2():
    k1 = 1.1
    k2 = 1
    b = 0.7

    while k2 <= 1000:
        run(k1, k2, b)
        k2 = k2 * 5


if __name__ == "__main__":
    # run(float(sys.argv[1]),
    #     float(sys.argv[2]),
    #     float(sys.argv[3]))
    grid_search_b()
