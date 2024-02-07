import sys

from table_lib import tsv_iter


def pattern_match(is_number_like, q_term, d_term):
    return is_number_like(q_term) and is_number_like(d_term)


def main():
    def is_number_like(term: str) -> bool:
        for i in range(10):
            if str(i) in term:
                return True

        return False

    for q_term, d_term, score in tsv_iter(sys.argv[1]):
        if pattern_match(is_number_like, q_term, d_term):
            print("\t".join([q_term, d_term, score]))


if __name__ == "__main__":
    main()
