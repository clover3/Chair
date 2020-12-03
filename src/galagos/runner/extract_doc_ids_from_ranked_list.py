import sys

from galagos.parse import parse_galago_ranked_list_with_space


def main():
    file_path = sys.argv[1]
    itr = open(file_path, "r")
    ranked_list = parse_galago_ranked_list_with_space(itr)
    for query, l in ranked_list.items():
        for e in l:
            print(e.doc_id)


if __name__ == "__main__":
    main()