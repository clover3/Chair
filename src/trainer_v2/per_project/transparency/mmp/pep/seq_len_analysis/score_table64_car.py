from trainer_v2.per_project.transparency.misc_common import read_term_pair_table
from trainer_v2.per_project.transparency.mmp.pep.seq_len_analysis.score_table32 import TermPairPredictionConfig, \
    predict_with_fixed_context_model_and_save


class PredictionConfig64_1(TermPairPredictionConfig):
    max_seq_length = 64
    num_classes = 1
    model_type = "bert-base-uncased"
    n_mask_prepad = 4
    n_mask_postpad = 8



def main():
    table_path = "output/mmp/car_dev1000_d_terms_pep_paired.txt"
    candidate_pairs = read_term_pair_table(table_path)
    num_items = len(candidate_pairs)
    model_path = "output/model/runs/mmp_pep10_point/model_20000"
    config = PredictionConfig64_1()
    log_path = f"output/mmp/tables/car_{config.max_seq_length}_{config.n_mask_postpad}.tsv"
    predict_with_fixed_context_model_and_save(
        config,
        model_path, log_path, candidate_pairs, 100, num_items)


if __name__ == "__main__":
    main()
