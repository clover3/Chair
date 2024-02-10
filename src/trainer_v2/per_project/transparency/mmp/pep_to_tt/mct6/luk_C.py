import sys

from trainer_v2.per_project.transparency.mmp.pep_to_tt.mct6.score_mct6 import build_mct6_config
from trainer_v2.per_project.transparency.mmp.retrieval_run.bm25t_luk import run_bm25t_luk


def main():
    model_name = sys.argv[1]
    step = int(sys.argv[2])
    dataset = "dev_C"
    conf = build_mct6_config(model_name, step)
    run_name = conf.job_name_base + "_" + dataset
    run_bm25t_luk(run_name, conf.table_save_path, dataset)


if __name__ == "__main__":
    main()