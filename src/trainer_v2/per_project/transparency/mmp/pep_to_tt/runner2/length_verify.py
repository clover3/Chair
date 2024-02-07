import sys

from data_generator.tokenizer_wo_tf import get_tokenizer
from tab_print import print_table
from table_lib import tsv_iter


def check_length(max_seq_length, qt_dt_iter):
    tokenizer = get_tokenizer()

    fail_list = []

    for q_term, d_term in qt_dt_iter:
        q_term_tokens = tokenizer.tokenize(q_term)
        d_term_tokens = tokenizer.tokenize(d_term)
        max_seg2_len = max_seq_length - 3 - len(q_term_tokens)
        if len(d_term_tokens) > max_seg2_len:
            fail_list.append((q_term, d_term))
            # print("Cut {} to {}".format(len(d_term_tokens), max_seg2_len))
    return fail_list


def main():
    max_seq_length = 16
    items = [(qt, dt) for qt, dt, _ in tsv_iter(sys.argv[1])]
    fail_list = check_length(max_seq_length, items)
    print_table(fail_list)
    print("Total of {} out of {}".format(len(fail_list), len(items)))


if __name__ == "__main__":
    main()
