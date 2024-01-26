from table_lib import tsv_iter

import sys
from omegaconf import OmegaConf


class PtoK:
    def __init__(self, table_path):
        self.d = {p_word: k_word for p_word, k_word in tsv_iter(table_path)}

    def convert(self, term):
        try:
            return self.d[term]
        except KeyError:
            return term


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)

    p_to_k = PtoK(conf.p_to_k_table)

    with open(conf.converted_path, "w") as f:
        for qt, dt in tsv_iter(conf.src_pair_path):
            f.write("{}\t{}\n".format(p_to_k.convert(qt), p_to_k.convert(dt)))




if __name__ == "__main__":
    main()