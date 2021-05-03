from dataset_specific.msmarco.common import at_working_dir


def main():
    f = open(at_working_dir("msmarco-docs.tsv"), "r", encoding="utf8")
    out_f = open(at_working_dir("msmarco-docs-offset.tsv"), "w", encoding="utf8")

    def log_offset(idx, offset):
        out_f.write("{}\t{}\n".format(idx, offset))

    idx = 0
    log_offset(idx, f.tell())

    line = f.readline()
    while line:
        idx += 1
        offset = f.tell()  # returns the location of the next line
        log_offset(idx, offset)
        line = f.readline()

        if idx % 1000 == 0:
            print(idx)

    ##

if __name__ == "__main__":
    main()