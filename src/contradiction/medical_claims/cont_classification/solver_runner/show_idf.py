import math

from list_lib import right
from misc_lib import get_second
from tab_print import print_table
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import get_bioclaim_retrieval_corpus
from trainer_v2.per_project.tli.qa_scorer.bm25_system import build_stats


def term_idf_factor(N, df):
    return math.log((N - df + 0.5) / (df + 0.5))


def main():
    split = "dev"
    question, claims = get_bioclaim_retrieval_corpus(split)
    df, cdf, avdl = build_stats(right(question))
    print("cdf", cdf)

    output = []
    for t, df_v in df.items():
        idf = term_idf_factor(cdf, df_v)
        output.append((t, df_v, idf))

    output.sort(key=get_second)
    print_table(output)


if __name__ == "__main__":
    main()