import csv
import sys


def modify_tsv(file_path, out_path, word_to_repeat):
    # Read the TSV file
    with open(file_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        rows = list(reader)

    # Replace the first column of each row with the specified word
    modified_rows = [[word_to_repeat] + row for row in rows]

    # Write the modified data to a new file or print it
    with open(out_path, 'w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        writer.writerows(modified_rows)





def main():
    # Example usage
    modify_tsv(sys.argv[1],
               sys.argv[2],
               "involved")


if __name__ == "__main__":
    main()