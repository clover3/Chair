

# TODO For q in queries with when
#               get d+, d-
#               Select term j in d+
#               Assign W[when,j] = 1
# Using W[when, j] = 1, rerank the queries.
#       W[when, j] = 0,
#       K=0


# Model inspector



N = NotImplemented
def select(query, d_pos, d_neg):
    attention = N
    idx_when = N # index of when in attention
    W = [] # array
    q_terms = []

    bert_score_pos = 1
    bert_score_neg = -1
    bm25_score_pos = 1
    bm25_score_neg = -1

    bert_decision = (bert_score_pos - bert_score_neg) > 1
    bm25_decision = (bm25_score_pos - bm25_score_neg) > 1
    # pass
    if bert_decision and not bm25_decision:
        for q_i in range(len(q_terms)):
            bm25_match = 1



        #
        # align_score = attention[idx_when] * gradient(attention[idx_when])
        # top_k = 3
        # for j in argsort(align_score)[:top_k]:
        #     pass
        #     W[j] = 1

    # Question: Are all top_k important?
    # Can we guess W[j] without encoding documents that contain j ?
    #   Maybe we can replace term j into j' by subtracting adding vectors.
    #   Then, we can compute only one transformer block on one token and its contexts to infer output
    #   #   #   #   #   #   #

