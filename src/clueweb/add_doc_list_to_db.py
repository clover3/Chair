import sys

from clueweb.clueweb_galago_db import add_doc_list_to_table


def main():
    file_path = sys.argv[1]
    save_name = sys.argv[2]

    doc_list = read_doc_list(file_path)

    add_doc_list_to_table(doc_list, save_name)


def read_doc_list(file_path):
    doc_list = []
    for line in open(file_path, "r"):
        doc_list.append(line.strip())
    return doc_list


if __name__ == "__main__":
    main()