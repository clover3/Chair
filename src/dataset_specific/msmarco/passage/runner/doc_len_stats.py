from cache import load_pickle_from
from cpath import output_path
from misc_lib import path_join, BinHistogram
from tab_print import tab_print_dict

# Result
# 128	8151704
# 256	688170
# 512	1938
# 1024	11
# (Based on BERT tokenization)


def main():
    dl_path = path_join(output_path, "mmp/bert_tokenized/dl")
    dl = load_pickle_from(dl_path)

    def bin_fn(n):
        if n < 128:
            return "128"
        if n < 256:
            return "256"
        if n < 512:
            return "512"
        if n < 1024:
            return "1024"

    bin = BinHistogram(bin_fn)
    for k, v in dl.items():
        bin.add(v)

    tab_print_dict(bin.counter)



if __name__ == "__main__":
    main()