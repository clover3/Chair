from cpath import output_path
from misc_lib import path_join
from table_lib import tsv_iter
from trainer_v2.per_project.transparency.mmp.bias.analyze import analyze_inner
from trainer_v2.per_project.transparency.mmp.bias.run_inf_w_keyword_swap_dev_split import load_car_bias_exp_resource_dev


def read_score_log():
    dir_path = path_join(output_path, "mmp", "bias", "score_log_dev")
    for i in range(0, 20):
        yield from tsv_iter(path_join(dir_path, f"{i}.tsv"))


def main():
    score_log = read_score_log()
    passages, query_list, term_list_set, term_list = load_car_bias_exp_resource_dev()
    analyze_inner(passages, query_list, score_log, term_list, term_list_set)



if __name__ == "__main__":
    main()
