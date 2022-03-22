import random
from typing import List, Dict, Tuple

from scipy import spatial
from sentence_transformers import SentenceTransformer

from contradiction.medical_claims.label_structure import AlamriLabel, PairedIndicesLabel
from contradiction.medical_claims.token_tagging.eval_analyze.online_eval import load_sbl_labels
from contradiction.medical_claims.token_tagging.problem_loader import AlamriProblem, \
    load_alamri_split
from datastore.cached_client import MemoryCachedClient
from list_lib import get_max_idx


def main():
    model = SentenceTransformer('sentence-transformers/bert-base-nli-mean-tokens')
    cache_client = MemoryCachedClient(model.encode, str, {})

    tag_type = "mismatch"
    split = "dev"
    random.seed(0)
    problems: List[AlamriProblem] = load_alamri_split(split)
    labels: List[AlamriLabel] = load_sbl_labels(split)

    labels_d: Dict[Tuple[int, int], PairedIndicesLabel] = {(l.group_no, l.inner_idx): l.label for l in labels}

    for p in problems[8:]:
        try:
            print("sent1: " + p.text1)
            print("sent2: " + p.text2)
            tokens1 = p.text1.split()
            tokens2 = p.text2.split()

            emb_list1 = cache_client.predict(tokens1)
            emb_list2 = cache_client.predict(tokens2)

            print(tokens1)
            print(tokens2)
            for i in range(len(tokens1)):
                score_list = []
                for j in range(len(tokens2)):
                    score = 1 - spatial.distance.cosine(emb_list1[i], emb_list2[j])
                    score_list.append(score)
                max_idx = get_max_idx(score_list)
                print("{} - {} ({})".format(tokens1[i], tokens2[max_idx], score_list[max_idx]))

        except KeyError:
            pass
        break

if __name__ == "__main__":
    main()
