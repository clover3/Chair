from cpath import output_path
from table_lib import print_positive_entry
from misc_lib import path_join

def main():
    file_read = path_join(output_path, "msmarco", "passage", "fidelity_2_1.tsv")
    print_positive_entry(file_read)


if __name__ == "__main__":
    main()