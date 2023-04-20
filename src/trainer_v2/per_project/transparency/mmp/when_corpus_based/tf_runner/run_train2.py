import sys
import tensorflow as tf

from trainer_v2.chair_logging import c_log
from trainer_v2.custom_loop.run_config2 import RunConfig2, get_run_config2
from trainer_v2.custom_loop.train_loop_helper import get_strategy_from_config
from trainer_v2.per_project.transparency.mmp.one_q_term_modeling import get_dataset, get_model, get_model2
from trainer_v2.per_project.transparency.mmp.when_corpus_based.when_bm25t import get_candidate_voca
from trainer_v2.train_util.arg_flags import flags_parser


def main(args):
    c_log.info(__file__)
    run_config: RunConfig2 = get_run_config2(args)
    run_config.print_info()
    voca_size = 699 + 1

    def build_dataset(input_files, is_for_training):
        return get_dataset(
            input_files, voca_size, run_config)

    candidate_voca = get_candidate_voca()
    when_id = candidate_voca['when']

    run_name = str(run_config.common_run_config.run_name)
    c_log.info("Run name: %s", run_name)

    strategy = get_strategy_from_config(run_config)
    with strategy.scope():
        train_dataset = build_dataset(run_config.dataset_config.train_files_path, True)
        if run_config.dataset_config.eval_files_path:
            eval_dataset = build_dataset(run_config.dataset_config.eval_files_path, False)
        else:
            eval_dataset = None
        c_log.info("Building model")
        model = get_model2(voca_size, when_id, run_config)
        model.evaluate(train_dataset)
        model.fit(train_dataset,
                  validation_data=eval_dataset,
                  epochs=2,
                  )
        model.save(run_config.train_config.model_save_path)
        c_log.info("Saved model at %s", run_config.train_config.model_save_path)


if __name__ == "__main__":
    args = flags_parser.parse_args(sys.argv[1:])
    main(args)


