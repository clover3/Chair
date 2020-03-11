from collections import Counter

from scipy.stats import stats

from arg.claim_building.clueweb12_B13_termstat import load_clueweb12_B13_termstat


def analyze(all_voca, doc_list, unigrams):
    # Do count
    cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, \
    clueweb_df, clueweb_tf, ctf_cont, ctf_ncont, \
    df_cont, df_ncont, tf_cont, tf_ncont = count_term_stat(doc_list, unigrams)

    # check hypothesis
    check_hypothesis(all_voca, cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, clueweb_df, clueweb_tf, ctf_cont,
                     ctf_ncont, df_cont, df_ncont, tf_cont, tf_ncont, unigrams)


def count_term_stat(doc_list, unigrams):
    # count term frequency
    # t \in unigrams
    # df(controversy, t),  df(t)
    # df_clueweb(controversy), df_clueweb(t)

    tf_cont = Counter()
    tf_ncont = Counter()
    ctf_cont = 0
    ctf_ncont = 0
    df_cont = Counter()
    df_ncont = Counter()
    cdf_cont = 0
    cdf_ncont = 0
    clueweb_tf, clueweb_df = load_clueweb12_B13_termstat()
    clueweb_ctf = sum(clueweb_tf.values())
    clueweb_cdf = max(clueweb_df.values()) + 100

    def get_tf(doc, t):
        s = 0
        for token in doc['tokens']:
            if token == t:
                s += 1
        return s

    def contain_controversy(doc):
        return 'controversy' in doc['tokens_set'] or 'controversial' in doc['tokens_set']

    def contain(doc, t):
        return t in doc['tokens_set']

    for doc in doc_list:
        current_doc_contain_controversy = contain_controversy(doc)
        for t in unigrams:
            if contain(doc, t):
                if current_doc_contain_controversy:
                    tf_cont[t] += get_tf(doc, t)
                    df_cont[t] += 1
                else:
                    tf_ncont[t] += get_tf(doc, t)
                    df_ncont[t] += 1

        if current_doc_contain_controversy:
            ctf_cont += len(doc['tokens'])
            cdf_cont += 1
        else:
            ctf_ncont += len(doc['tokens'])
            cdf_ncont += 1
    return cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, clueweb_df, clueweb_tf, ctf_cont, ctf_ncont, df_cont, df_ncont, tf_cont, tf_ncont


def check_hypothesis(all_voca, cdf_cont, cdf_ncont, clueweb_cdf, clueweb_ctf, clueweb_df, clueweb_tf, ctf_cont,
                     ctf_ncont, df_cont, df_ncont, tf_cont, tf_ncont, unigrams):
    hypo1 = []
    hypo1_1 = []
    hypo2_1 = []
    hypo2_2 = []
    for t in unigrams:
        if t not in all_voca:
            print("Warning {} is never observed".format(t))
            continue

        # Hypothesis 1 : P(t|controversy,R) > P(t| !controversy,R)
        # Hypothesis 2 : P(t|R) > P(t|BG)

        p1 = tf_cont[t] / ctf_cont
        p2 = tf_ncont[t] / ctf_ncont
        hypo1.append((t, (p1, p2)))
        p1 = df_cont[t] / cdf_cont
        p2 = df_ncont[t] / cdf_ncont
        hypo1_1.append((t, (p1, p2)))

        p1 = (tf_cont[t] + tf_ncont[t]) / (ctf_cont + ctf_ncont)
        if t not in clueweb_df:
            print("warning {} not in clueweb voca".format(t))
            continue

        p2 = clueweb_tf[t] / clueweb_ctf
        hypo2_1.append((t, (p1, p2)))

        p1 = (df_cont[t] + df_ncont[t]) / (cdf_cont + cdf_ncont)
        p2 = clueweb_df[t] / clueweb_cdf
        hypo2_2.append((t, (p1, p2)))
    todo = [(hypo1, "Hypothesis 1 : P(t|controversy,R) > P(t| !controversy,R)"),
            (hypo1_1, "Hypothesis 1 : P(t|controversy,R) > P(t| !controversy,R) by binary model"),
            (hypo2_1, "Hypothesis 2 : P(t|R) > P(t|BG)"),
            (hypo2_2, "Hypothesis 2 : P(t|R) > P(t|BG) by binary model"),
            ]
    for hypo, desc in todo:
        print(desc)
        terms, pairs = zip(*hypo)
        p1_list, p2_list = zip(*pairs)
        _, p = stats.ttest_rel(p1_list, p2_list)
        print(p)
        for term, pair in hypo:
            p1, p2 = pair
            print(term, "{0:.4f} {1:.4f}".format(p1, p2))