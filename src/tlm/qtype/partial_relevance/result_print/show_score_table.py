from tab_print import print_table
from tlm.qtype.partial_relevance.calc_avg import calc_count_avg


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
        avg, n_total, n_valid = calc_count_avg(run_name)
        row = [run_name, n_total, n_valid, avg]
        rows.append(row)

    print_table(rows)


def main():
    print_avg()


if __name__ == "__main__":
    main()