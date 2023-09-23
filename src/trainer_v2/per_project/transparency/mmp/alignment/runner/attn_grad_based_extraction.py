import sys

from scratch.code2023.omegaconf_prac import load_tf_predict_config
from trainer_v2.per_project.config_util import strategy_with_pu_config
from trainer_v2.per_project.transparency.mmp.alignment.alignment_predictor import compute_alignment_any_pair
from trainer_v2.per_project.transparency.mmp.alignment.grad_extractor import extract_save_align
from trainer_v2.per_project.transparency.mmp.dev_analysis.when_term_frequency import enum_when_corpus


def main():
    config = load_tf_predict_config(sys.argv[1])
    compute_alignment_fn = compute_alignment_any_pair

    def enum_qd_pairs():
        for query, doc_pos, doc_neg in enum_when_corpus():
            yield query, doc_pos
            yield query, doc_neg

    num_record = 13220
    qd_itr = enum_qd_pairs()
    save_path = config.save_path
    strategy = strategy_with_pu_config(config.pu_config)
    extract_save_align(compute_alignment_fn, qd_itr, strategy, save_path,
                       num_record, config.model_load_config.model_path,
                       config.batch_size)


if __name__ == "__main__":
    main()

