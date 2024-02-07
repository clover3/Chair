import sys

from table_lib import tsv_iter


def main():
    def contain_a_hat(term: str) -> bool:
        return "â" not in term

    pattern_match = contain_a_hat

    for q_term, d_term, score in tsv_iter(sys.argv[1]):
        if pattern_match(q_term) and pattern_match(d_term):
            print("\t".join([q_term, d_term, score]))


if __name__ == "__main__":
    main()
