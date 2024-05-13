import sys

from trainer_v2.per_project.transparency.misc_common import read_term_pair_table_w_score, save_term_pair_scores


def main():
    table = read_term_pair_table_w_score(sys.argv[1])

    def get_key(e):
        return e[0], -e[2]

    table = [e for e in table if e[2] > 0.01]
    table.sort(key=get_key)
    save_term_pair_scores(table, sys.argv[2])


if __name__ == "__main__":
    main()