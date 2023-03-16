from cache import load_from_pickle


def load_msmarco_passage_term_stat():
    df = load_from_pickle("msmarco_passage_df")
    cdf = 8841823
    return cdf, df

