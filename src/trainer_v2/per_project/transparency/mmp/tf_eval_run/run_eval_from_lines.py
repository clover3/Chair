import os.path
import sys

from omegaconf import OmegaConf

from trainer_v2.chair_logging import c_log
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import \
    run_build_ranked_list_from_line_scores_and_eval


def main():
    dataset_conf_path = sys.argv[1]
    model_path = sys.argv[2]
    run_name = os.path.basename(model_path)
    conf = OmegaConf.create({
        'run_name': run_name,
        'dataset_conf_path': dataset_conf_path,
        'outer_batch_size': 256,
    })
    run_build_ranked_list_from_line_scores_and_eval(conf)


if __name__ == "__main__":
    main()
