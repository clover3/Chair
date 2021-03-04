from misc_lib import BinHistogram
from tab_print import print_table


def summarize_doc_length(text_iter):

    num_doc = 100000
    # bh_char_len = BinHistogram(id)
    bh_token_len = BinHistogram(lambda x: x)

    for idx, text in enumerate(text_iter):
        num_char = len(text)
        tokens = text.split()
        # bh_char_len.add(num_char)
        bh_token_len.add(len(tokens))
        if num_doc == idx + 1:
            break

    # bh_char_len.counter.keys()

    interesting_points = [0, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999]

    def show_histogram(bh):
        interesting_point_cursor = 0
        keys = list(bh.counter.keys())
        keys.sort()
        portion_acc = 0
        rows = []
        for k in keys:
            portion = bh.counter[k] / num_doc
            portion_acc += portion

            # interesting_cut = interesting_points[interesting_point_cursor]
            # if portion_acc > interesting_cut:
            row = [k, bh.counter[k], portion, portion_acc]
            rows.append(row)
            if portion_acc > 0.999:
                break
        print_table(rows)

    show_histogram(bh_token_len)