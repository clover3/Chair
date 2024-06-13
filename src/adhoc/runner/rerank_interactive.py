import sys

from adhoc.resource.scorer_loader import get_rerank_scorer
from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
from trainer_v2.chair_logging import c_log


def main():
    c_log.info(__file__)
    method = "contriever"
    score_fn = get_rerank_scorer(method).score_fn

    while True:
        query = input("Enter query : ")
        doc = input("Enter document: ")
        ret = score_fn([(query, doc)])[0]
        print(ret)


if __name__ == "__main__":
    main()
