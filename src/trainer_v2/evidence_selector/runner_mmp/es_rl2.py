import logging
import os
import sys
from cpath import get_bert_config_path
from port_info import LOCAL_DECISION_PORT
from taskman_client.wrapper3 import report_run3
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig512_2, ModelConfig256_2
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.seq_pred import SeqPrediction
from trainer_v2.custom_loop.per_task.rl_trainer import PolicyGradientTrainer
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop import tf_run

from trainer_v2.evidence_selector.environment import PEPEnvironment, PEPClient
from trainer_v2.evidence_selector.environment_qd import ConcatMaskStrategyQD, get_pe_for_qd_mae, get_pe_for_qd_inner
from trainer_v2.evidence_selector.evidence_scoring import mean_absolute_error
from trainer_v2.evidence_selector.runner_mmp.dataset_fn import build_state_dataset_fn
from trainer_v2.evidence_selector.seq_pred_policy_gradient import SeqPredREINFORCE
from trainer_v2.evidence_selector.policy_function_for_evidence_selector import SequenceLabelPolicyFunction
from trainer_v2.tf_misc_helper import SummaryWriterWrap
from trainer_v2.train_util.arg_flags import flags_parser
from typing import List, Iterable, Callable, Dict, Tuple, Set

@report_run3
def main(args):
    c_log.info(__file__)
    src_model_config = ModelConfig512_2()
    model_config = ModelConfig256_2()
    concat_mask = ConcatMaskStrategyQD()
    tolerance = 0.05
    density_weight = 1
    c_log.setLevel(logging.DEBUG)

    def get_pe_factory(base_pred, rep_pred, action, state) -> [[], Callable]:
        return get_pe_for_qd_inner(
            base_pred, rep_pred, action, state, mean_absolute_error, tolerance, density_weight)

    server = "localhost"
    if "PEP_SERVER" in os.environ:
        server = os.environ["PEP_SERVER"]
    c_log.info("PEP_SERVER: {}".format(server))
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    pep_client = PEPClient(server, LOCAL_DECISION_PORT)
    pep_env = PEPEnvironment(
        pep_client, get_pe_factory, concat_mask)

    task_model = SeqPrediction()
    build_state_dataset = build_state_dataset_fn(run_config, src_model_config)
    window_length = model_config.max_seq_length
    summary_writer = SummaryWriterWrap(run_config.common_run_config.run_name)

    reinforce = SeqPredREINFORCE(
        window_length,
        build_state_dataset,
        run_config.common_run_config.batch_size,
        pep_env.get_item_results,
        concat_mask,
        summary_writer
    )
    num_sample = 100

    def policy_func_factory(model):
        return SequenceLabelPolicyFunction(model, num_sample, "second")

    # Trainer for TF policy model update
    trainer: PolicyGradientTrainer = PolicyGradientTrainer(
        bert_params,
        model_config,
        run_config,
        task_model,
        policy_func_factory,
        reinforce
    )
    tf_run(run_config, trainer, trainer.build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)

