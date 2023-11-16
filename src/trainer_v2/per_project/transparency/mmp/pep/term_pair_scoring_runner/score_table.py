import sys

from omegaconf import OmegaConf

from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import read_term_pair_table
from trainer_v2.per_project.transparency.mmp.pep.inf_helper import predict_with_fixed_context_model


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    candidate_pairs = read_term_pair_table(conf.table_path)
    num_items = len(candidate_pairs)
    model_path = conf.model_path
    log_path = conf.save_path
    predict_with_fixed_context_model(model_path, log_path, candidate_pairs, 100, num_items)


if __name__ == "__main__":
    main()
