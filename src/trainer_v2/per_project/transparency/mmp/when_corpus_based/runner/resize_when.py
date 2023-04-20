from dataset_specific.msmarco.passage.passage_resource_loader import enum_all_when_corpus, enum_grouped
from cpath import output_path
from misc_lib import path_join, TELI


def main():
    maybe_num_group = 16445
    size_per_block = 1000 # 1000 query per block
    itr = enum_all_when_corpus()
    itr2 = enum_grouped(itr)
    itr2 = TELI(itr2, maybe_num_group)
    file_idx = 0
    def get_new_file():
        nonlocal file_idx
        save_path = path_join(output_path, "msmarco", "passage", "when_full_re", str(file_idx))
        f = open(save_path, "w")
        file_idx += 1
        return f

    f = get_new_file()
    n_print = 0
    for group in itr2:
        if n_print >= size_per_block:
            f = get_new_file()
            n_print = 0

        for t in group:
            f.write("\t".join(t) + "\n")
        n_print += 1


if __name__ == "__main__":
    main()

