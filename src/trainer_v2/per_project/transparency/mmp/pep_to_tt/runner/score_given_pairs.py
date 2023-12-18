import sys
from typing import Iterable

from omegaconf import OmegaConf

from table_lib import tsv_iter
from taskman_client.wrapper3 import JobContext
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_common import predict_pairs_save
from trainer_v2.per_project.transparency.mmp.pep_to_tt.inf_helper import PEP_TT_Inference
from trainer_v2.per_project.transparency.mmp.pep_to_tt.pep_tt_modeling import PEP_TT_ModelConfig


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    step = sys.argv[2]

    qd_pairs: Iterable[tuple[str, str]] = [(qt, dt) for qt, dt, _ in tsv_iter(conf.pair_path)]
    model_path = conf.model_path.format(step)
    save_path = conf.table_save_path.format(step)
    job_name = conf.run_name

    model_config = PEP_TT_ModelConfig()
    inf_helper = PEP_TT_Inference(
        model_config,
        model_path,
        model_type=conf.model_type)

    predict_term_pairs_fn = inf_helper.score_fn
    with JobContext(job_name):
        predict_pairs_save(
            predict_term_pairs_fn, qd_pairs,
            save_path, outer_batch_size=100)


if __name__ == "__main__":
    main()