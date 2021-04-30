from collections import Counter

from arg.perspectives.pc_tokenizer import PCTokenizer
from cache import save_to_pickle
from dataset_specific.msmarco.common import at_working_dir
from misc_lib import Averager, TimeEstimator


def get_df():
    doc_f = open(at_working_dir("msmarco-docs.tsv"), encoding="utf8")
    tokenizer = PCTokenizer()
    line_itr = doc_f
    df = Counter()

    n_doc = 0
    n_tokens = 0###
    n_record = 3213835
    skip_rate = 10
    time_estimator = TimeEstimator(n_record, "msmarco", 2000)

    for idx, line in enumerate(line_itr):
        time_estimator.tick()
        if idx % skip_rate == 0:
            pass
        else:
            continue

        doc_id, url, title, body = line.split("\t")
        content = title + " " + body
        tokens = tokenizer.tokenize_stem(content)
        dl = len(tokens)
        n_tokens += dl
        n_doc += 1
        for t in set(tokens):
            df[t] += 1

    print("collection size:", n_tokens)
    print("num docs", n_doc)
    print("avg dl", n_tokens / n_doc)
    save_to_pickle(df, "mmd_df_{}".format(skip_rate))


if __name__ == "__main__":
    get_df()