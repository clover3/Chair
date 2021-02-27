from misc_lib import BinHistogram


def summarize_doc_length(text_iter):
    def cent(n):
        return int(n / 100)

    num_doc = 100000
    bh_char_len = BinHistogram(cent)
    bh_token_len = BinHistogram(cent)

    for idx, text in enumerate(text_iter):
        num_char = len(text)
        tokens = text.split()
        bh_char_len.add(num_char)
        bh_token_len.add(len(tokens))
        if num_doc == idx + 1:
            break

    bh_char_len.counter.keys()

    interesting_points = [0, 0.5, 0.7, 0.9, 0.95, 0.99, 0.995, 0.999]

    def show_histogram(bh):
        interesting_point_cursor = 0
        keys = list(bh.counter.keys())
        keys.sort()
        portion_acc = 0
        for k in keys:
            portion = bh.counter[k] / num_doc
            portion_acc += portion

            interesting_cut = interesting_points[interesting_point_cursor]
            if portion_acc > interesting_cut:
                print("{0} {1} {2:3f} {3:.4f}".format(k, bh.counter[k], portion, portion_acc))
                interesting_point_cursor += 1

            if interesting_point_cursor >= len(interesting_points):
                break

    show_histogram(bh_char_len)
    show_histogram(bh_token_len)