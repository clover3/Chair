


# For each item in Data:
#     print(a)
#
#
#
#
# Alignment

When EM algorithms are used to estimate translation model, there is no observable evidences for estimating
alignments and matching (translation)

When we are building a global proxy for neural ranking model, we can use some information

When cross-encoders are used attention can be used to infer the matching.
- High attention scored pairs are likely to indicates relevance.
-
QT Query term space
QTS Query term space - stemmed
DT Document term space
DTS Document term space - stemmed
-
Start with QTS/DTS.
We have a table T, which is an identity vector.

Let i and j denote a terms in QTS.

We want to know if T[i,j] should be zero or not.

1) Co-occurrence of i and j
2) Co-occurrence of i and j in relevant documents
3)


Q: Query
D: Document
A: Alignment

Input: List[Tuple[Q,D,A]]

Let T be a translation table of size |V|*|V| and TF be a list of term frequencies with size |V|

Modified term frequencies are given by
TF' = T * TF
Let TF'[q_i] indicate a row that correspond to the query term q_i.

The score for q_i is given as
   Score(q,d) = \sum_i sum(TF[q_i])

1) Optimize full table T with SGD.


For query in TrainingData:
    top1000 document
    pos
    neg


    Use all 270K voca?
    1) Use voca that appears for query



2) For each Q,D


# i_q, i_d are indices for query and document
For (i_q, i_d), score in Alignment:
   q_i = Q[i_q] # query term
   d_i = D[i_d] # document term
   Observation(q_i, d_i).append((score))
   * P(q_i|d_i) = Q[i_q] * D[i_d]

What if no align?

W[q_i, d_i] += 1





