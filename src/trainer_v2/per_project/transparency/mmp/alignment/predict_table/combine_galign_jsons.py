import sys

from omegaconf import OmegaConf

from data_generator.tokenizer_wo_tf import get_tokenizer
from trainer_v2.per_project.transparency.misc_common import save_tsv
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.combine_inference import \
    filter_sharp_sharp_subword, convert_term_pair_candidate_ids_to_terms
from trainer_v2.per_project.transparency.mmp.alignment.predict_table.combine_inference import \
    load_term_pair_candidate_over_100_jobs
from typing import List, Iterable, Callable, Dict, Tuple, Set


def main():
    run_conf_path = sys.argv[1]
    run_conf = OmegaConf.load(run_conf_path)
    dir_path = run_conf.json_dir
    candidate_pair_save_path = run_conf.candidate_pair_save_path
    items = load_term_pair_candidate_over_100_jobs(dir_path)
    tokenizer = get_tokenizer()
    all_candidates: List[Tuple[str, str, float]]\
        = convert_term_pair_candidate_ids_to_terms(tokenizer, items)
    selected = filter_sharp_sharp_subword(all_candidates)
    save_tsv(selected, candidate_pair_save_path)



if __name__ == "__main__":
    main()