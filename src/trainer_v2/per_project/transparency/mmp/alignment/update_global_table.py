
"""


For (q, d+, d-) in Corpus:
   Loss = 1 - (Score(q,d+) - Score(q,d-))
   If Loss > 0:
      for positive term in d+ compute gradient.
   grad = -dL/dw

   for term in global_table:
      if term in d:



"""

def bm25_grad(qtw, tf, dl, avgdl):
    pass


# Candidate terms: Terms that are in global table
# TODO how to enum meaningful pairs
#     positive document i,
#     set of false positive document J
#     set of true negative document J2
#     the candidate terms in positive document i gets positive gradient
#     the candidate term in false positive documents get negative gradient.
#


def update_global_table():
    itr = 0
    for q, d_pos, d_neg in itr:
        pass

        # TODO: Update POS entry.
        aligned_terms = {}
        for term in aligned_terms:
            gradient = bm25_grad()
            terms = NotImplemented

