import sys

from tab_print import print_table
from tlm.data_gen.adhoc_datagen import MultiWindowTokenCount
from tlm.data_gen.robust_gen.robust_generators import RobustTrainTextSize
from tlm.robust.load import robust_query_intervals


def count_actual_tokens(src_window_size, total_sequence_length):
    encoder = MultiWindowTokenCount(src_window_size, total_sequence_length)
    counter_fn = RobustTrainTextSize(encoder, total_sequence_length, "desc")
    all_cnt = []
    for job_id in range(5):
        st, ed = robust_query_intervals[job_id]
        query_list = [str(i) for i in range(st, ed + 1)]
        cnt_list = counter_fn.count(query_list)
        all_cnt.extend(cnt_list)
    return all_cnt


# 450 words per 512 window
def main():
    ##
    def get_num_tokens(src_window_size, n_window):
        total_sequence_length = src_window_size * n_window
        all_cnt = count_actual_tokens(src_window_size, total_sequence_length)
        table = [("src_window_size", src_window_size),
                 ("n_window", n_window),
                 ("sum", sum(all_cnt))
                ]
        print_table(table)

    src_window_size = int(sys.argv[1])
    n_window = int(sys.argv[2])
    get_num_tokens(src_window_size,
                   n_window)


if __name__ == "__main__":
    main()

