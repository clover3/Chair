import argparse
import sys

from cpath import at_output_dir
from tlm.data_gen.robust_gen.select_supervision.read_score import generate_selected_training_data_for_many_runs

arg_parser = argparse.ArgumentParser(description='')


def main():
    target_data_idx = int(sys.argv[1])
    info_dir = "/mnt/disks/disk100/data_info/robust_w_data_id_desc_info_pickle/"
    max_seq_length = 512
    score_and_save_dir = []
    base_model_name = "robust_3A"
    for split_idx in range(5):
        for repeat_idx in range(5):
            if target_data_idx == split_idx:
                pass
            else:
                score_dir_name = "seg_score_{}_{}_{}".format(base_model_name, split_idx, repeat_idx)
                score_dir_path = at_output_dir("robust", score_dir_name)
                save_dir_path = at_output_dir("robust_seg_sel", score_dir_name)
                score_and_save_dir.append((score_dir_path, save_dir_path))

    generate_selected_training_data_for_many_runs(
        target_data_idx,
        info_dir,
        max_seq_length,
        score_and_save_dir
    )


if __name__ == "__main__":
    main()