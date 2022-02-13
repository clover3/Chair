from tab_print import print_table
from tlm.qtype.partial_relevance.calc_avg import calc_count_avg, load_eval_result_r, load_eval_result_b


def print_avg():
    run_name_list = [
        "dev_attn_perturbation_erasure",
        "dev_attn_perturbation_partial_relevant",
        "dev_gradient_erasure",
        "dev_gradient_partial_relevant",
        "dev_random_erasure",
        "dev_random_partial_relevant",
    ]
    head = ["run_name", "n_total", "n_valid", "avg"]
    rows = [head]
    for run_name in run_name_list:
        eval_res = load_eval_result_r(run_name)
        avg, n_total, n_valid = calc_count_avg(eval_res)
        row = [run_name, n_total, n_valid, avg]
        rows.append(row)

    print_table(rows)


def print_avg2():
    dataset = "dev_sent"
    method_list = ["exact_match", "gradient", "random", "exact_match_noise0.1"]
    # method_list = ["exact_match", "random", "exact_match_noise0.1"]
    policy_name_list = ["attn", "ps_replace_precision", "ps_replace_recall",
                        "ps_deletion_precision", "ps_deletion_recall"]

    def get_run_name(method, policy_name):
        return "{}_{}_{}".format(dataset, method, policy_name)

    rows = []
    for policy in policy_name_list:
        for method in method_list:
            run_name = get_run_name(method, policy)
            eval_res = load_eval_result_b(run_name)
            avg, n_total, n_valid = calc_count_avg(eval_res)
            assert n_valid == n_valid
            row = [policy, method, avg, n_total]
            rows.append(row)
    print_table(rows)


def main():
    print_avg2()


if __name__ == "__main__":
    main()