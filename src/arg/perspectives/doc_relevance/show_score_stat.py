import os
from collections import Counter
from typing import List, Dict

from arg.perspectives.doc_relevance.common import load_doc_scores
from arg.perspectives.load import d_n_claims_per_split2
from cpath import output_path


def main():
    split = "dev"
    num_jobs = d_n_claims_per_split2[split]
    scores: Dict[int, List] = load_doc_scores(os.path.join(output_path, "pc_qk_1k_{}_score".format(split)), num_jobs)
    threshold = 0.9
    counter = Counter()
    for cid, doc_scores in scores.items():
        rel_docs = list([doc for doc, score in doc_scores if score > threshold])

        cnt = len(rel_docs)
        if cnt == 0:
            counter["0"] += 1
        elif cnt < 10:
            counter["1~9"] += 1
        elif cnt < 100:
            counter["10~99"] += 1
        else:
            counter["99<"] += 1

    for key, num in counter.items():
        print(key, num)


if __name__ == "__main__":
    main()