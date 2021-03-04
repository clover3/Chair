from exec_lib import run_func_with_config
from tlm.data_gen.robust_gen.select_supervision.read_score import generate_selected_training_data_loop


def main(config):
    score_dir = config['score_dir_path']  # "all_seg_score_robust_3A_4_4"
    max_seq_length = config['max_seq_length']
    info_path = config['info_path']
    save_dir = config['save_dir']
    split_no = config['split_no']

    generate_selected_training_data_loop(split_no, score_dir, info_path, max_seq_length, save_dir)


if __name__ == "__main__":
    run_func_with_config(main)
