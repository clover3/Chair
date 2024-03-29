import os
import sys
from cpath import get_bert_config_path
from port_info import LOCAL_DECISION_PORT
from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.definitions import ModelConfig600_2, ModelConfig300_2
from trainer_v2.custom_loop.modeling_common.bert_common import load_bert_config
from trainer_v2.custom_loop.neural_network_def.seq_pred import SeqPrediction
from trainer_v2.custom_loop.per_task.rl_trainer import PolicyGradientTrainer
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2_nli
from trainer_v2.custom_loop.train_loop import tf_run

from trainer_v2.evidence_selector.environment import PEInfo, PEPEnvironment, PEPClient
from trainer_v2.evidence_selector.enviroment_nli import get_pe_info_nli, PEPClientNLI, ConcatMaskStrategyNLI
from trainer_v2.evidence_selector.es_rl_dataset_helper import build_state_dataset_fn
from trainer_v2.evidence_selector.seq_pred_policy_gradient import SeqPredREINFORCE
from trainer_v2.evidence_selector.policy_function_for_evidence_selector import SequenceLabelPolicyFunction
from trainer_v2.train_util.arg_flags import flags_parser




def main(args):
    c_log.info("Start Train es_rl_train.py")
    src_model_config = ModelConfig600_2()
    model_config = ModelConfig300_2()
    concat_mask = ConcatMaskStrategyNLI()
    get_partial_evidence_info_fn = get_pe_info_nli

    server = "localhost"
    if "PEP_SERVER" in os.environ:
        server = os.environ["PEP_SERVER"]
    c_log.info("PEP_SERVER: {}".format(server))
    run_config: RunConfig2 = get_run_config2_nli(args)
    run_config.print_info()

    bert_params = load_bert_config(get_bert_config_path())
    pep_client = PEPClient(server, LOCAL_DECISION_PORT)
    pep_env = PEPEnvironment(
        pep_client, get_partial_evidence_info_fn, concat_mask)

    task_model = SeqPrediction()
    build_state_dataset = build_state_dataset_fn(run_config, src_model_config)
    window_length = model_config.max_seq_length

    reinforce = SeqPredREINFORCE(
        window_length,
        build_state_dataset,
        run_config.common_run_config.batch_size,
        pep_env.get_item_results,
        concat_mask
    )

    trainer: PolicyGradientTrainer = PolicyGradientTrainer(
        bert_params,
        model_config,
        run_config,
        task_model,
        SequenceLabelPolicyFunction,
        reinforce
    )
    tf_run(run_config, trainer, trainer.build_dataset)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


