from cpath import output_path
from table_lib import tsv_iter
from misc_lib import path_join

def main():
    file_read = path_join(output_path, "msmarco", "passage", "fidelity_2_1.tsv")

    for e in tsv_iter(file_read):
        score = float(e[2])
        if score > 0:
            print("\t".join(e))

    return NotImplemented


if __name__ == "__main__":
    main()