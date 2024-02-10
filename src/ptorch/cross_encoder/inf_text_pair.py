import sys

from ptorch.cross_encoder.get_ce_msmarco_mini_lm import get_ce_msmarco_mini_lm_score_fn
from cpath import output_path
from misc_lib import path_join, batch_iter_from_entry_iter
from table_lib import tsv_iter
from trainer_v2.chair_logging import c_log


def main():
    score_fn = get_ce_msmarco_mini_lm_score_fn()
    text_save_dir = path_join(output_path, "msmarco", "passage", "mmp1_attn_text")
    job_no = int(sys.argv[1])
    text_save_path = path_join(text_save_dir, f"{job_no}.txt")

    score_save_dir = path_join(output_path, "msmarco", "passage", "mmp1_attn_ce_mini_scores")
    score_save_path = path_join(score_save_dir, f"{job_no}.txt")
    f = open(score_save_path, "w")

    itr = tsv_iter(text_save_path)
    c_log.info("Start")
    for batch in batch_iter_from_entry_iter(itr, 1000):
        c_log.info("Inference......")
        scores = score_fn(batch)
        for s in scores:
            f.write(str(s) + "\n")


if __name__ == "__main__":
    main()