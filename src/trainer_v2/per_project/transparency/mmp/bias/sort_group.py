from list_lib import left
from tab_print import print_table
from trainer_v2.per_project.transparency.misc_common import load_tsv
from cpath import output_path
from misc_lib import path_join





def main():
    in_path = path_join(output_path, "mmp", "analysis", "good.txt")
    entries = load_tsv(in_path)

    # def get_key(pair):
    #     term, score = pair
    #     return len(term), term
    #
    # entries.sort(key=get_key)
    print(" ".join(left(entries)[:100]))
    # print_table(entries)


if __name__ == "__main__":
    main()