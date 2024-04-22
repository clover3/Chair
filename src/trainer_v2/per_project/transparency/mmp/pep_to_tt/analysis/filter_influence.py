import os
from math import log

from adhoc.other.bm25_retriever_helper import get_bm25_stats_from_conf
import sys
from omegaconf import OmegaConf

from table_lib import tsv_iter
from trainer_v2.per_project.transparency.misc_common import save_tsv


def main():
    conf_path = sys.argv[1]
    conf = OmegaConf.load(conf_path)
    avdl, cdf, df_d, _dl = get_bm25_stats_from_conf(conf,)
    N = cdf
    t = 1e10
    pattern_name = "influence_filtered"
    input_file = sys.argv[2]
    out_row = []
    file_name, file_extension = os.path.splitext(input_file)
    save_path = f"{file_name}_{pattern_name}{file_extension}"


    for q_term, d_term, score in tsv_iter(input_file):
        qt_df = df_d[q_term]
        assert N >= qt_df
        arg = (N-qt_df+0.5)/(qt_df + 0.5) + 1
        idf = log(arg)

        influence = idf * df_d[q_term] * df_d[d_term]
        if influence > t:
            out_row.append([q_term, d_term, score])

    save_tsv(out_row, save_path)



if __name__ == "__main__":
    main()