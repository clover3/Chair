import logging
import sys

from misc_lib import path_join
from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.retrieval_run.bm25t_luk import run_bm25t_luk_trec_dl19


def main():
    c_log.setLevel(logging.INFO)
    model_name = sys.argv[1]
    step = sys.argv[2]

    table_path = path_join("output", "mmp" , "tables", f"mtc4_{model_name}_{step}.tsv")
    run_name = f"mtc4_{model_name}_{step}_luk"
    run_bm25t_luk_trec_dl19(run_name, table_path)


if __name__ == "__main__":
    main()
