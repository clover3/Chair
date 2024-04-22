import sys

from adhoc.resource.scorer_loader import get_rerank_scorer
from misc_lib import exist_or_mkdir
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.misc_common import save_tsv


def main():
    c_log.info(__file__)
    for method in ["ce", "splade", "contriever", "contriever-msmarco", "tas_b"]:
        print(method)
        scorer = get_rerank_scorer(method)
        score_fn = scorer.score_fn
        todo = list(range(2000, 2025))
        exist_or_mkdir("output/mmp/bias/temporal")
        score_log_path = f"output/mmp/bias/temporal/{method}.tsv"
        scores = []
        for year in todo:
            query = "when did north carolina join ifta"
            doc = f"{year} north carolina join ifta "
            ret = score_fn([(query, doc)])
            if type(ret) == float:
                pass
            else:
                ret = ret[0]
            scores.append(ret)

        scores = map(str, scores)
        f = open(score_log_path, "w")
        for s in scores:
            f.write(f"{s}\n")


if __name__ == "__main__":
    main()
