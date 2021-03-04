import sys


def main():
    cuts = list(range(0, 1000, 100)) + list(range(1000, 3000, 200)) + list(range(3000, 10000, 1000))
    cut_idx = 0
    rows = []
    for line in open(sys.argv[1], "r"):
        n_word, n_docs, portion, portion_acc = line.split()
        row = n_word, n_docs, portion, portion_acc
        rows.append(row)


    for row_idx, row in enumerate(rows):
        n_word = row[0]
        n_word, n_docs, portion, portion_acc = row
        n_word_next = rows[row_idx+1][0]
        if int(n_word) <= cuts[cut_idx] < int(n_word_next):
            print("({} , {}),".format(cuts[cut_idx], portion_acc))
            cut_idx += 1
            if cut_idx >= len(cuts):
                break



##


if __name__ == "__main__":
    main()