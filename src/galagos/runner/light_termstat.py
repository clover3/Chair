import sys

from galagos.parse import load_term_stat


def main():
    tf, df = load_term_stat(sys.argv[1])

    fout = open(sys.argv[2], "w")
    for key, cnt in df.items():
        if cnt > 10:
            fout.write("{}\t{}\t{}\n".format(key, tf[key], df[key]))

    fout.close()



if __name__ == "__main__":
    main()