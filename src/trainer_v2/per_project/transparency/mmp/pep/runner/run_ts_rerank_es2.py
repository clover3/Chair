import sys

from omegaconf import OmegaConf

from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper3 import JobContext
from trainer_v2.chair_logging import c_log
from trainer_v2.evidence_selector.environment_qd import ConcatMaskStrategyQD
from trainer_v2.per_project.transparency.mmp.pep.pep_rerank_w_es import get_pep_scorer_es, get_hide_input_ids, \
    HideStrategyMinClip
from trainer_v2.per_project.transparency.mmp.eval_helper.rerank_w_conf import run_rerank_with_conf_common
from trainer_v2.train_util.get_tpu_strategy import get_strategy
from typing import List, Iterable, Callable, Dict, Tuple, Set
import numpy as np


def main():
    c_log.info(__file__)
    maks_strategy = ConcatMaskStrategyQD()

    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    mask_id = 0
    delete_portion = float(conf.delete_portion)
    hide = HideStrategyMinClip(maks_strategy, delete_portion, mask_id)

    def get_scorer_fn(conf):
        return get_pep_scorer_es(conf, hide.hide_input_ids)

    with JobContext(conf.run_name):
        strategy = get_strategy()
        with strategy.scope():
            run_rerank_with_conf_common(conf, get_scorer_fn)

    del_rate = hide.delete_rate.get_average()
    proxy = get_task_manager_proxy()
    proxy.report_number(conf.run_name, del_rate, "", "del_rate")


if __name__ == "__main__":
    main()
