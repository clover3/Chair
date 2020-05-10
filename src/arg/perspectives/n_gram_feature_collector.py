import os
import pickle
from collections import Counter
from typing import List, Dict, NamedTuple, Iterable

import nltk

from arg.perspectives.basic_analysis import PerspectiveCandidate
from arg.perspectives.select_paragraph_perspective import ParagraphClaimPersFeature
from arg.pf_common.base import ScoreParagraph
from data_generator.job_runner import sydney_working_dir
from list_lib import lmap
from misc_lib import exist_or_mkdir


class PCNGramFeature(NamedTuple):
    claim_pers: PerspectiveCandidate
    n_grams: Dict[int, Counter]


class PCVectorFeature(NamedTuple):
    claim_pers: PerspectiveCandidate
    vector: List


def tokenlist_hash(str_list: List[str]) -> int:
    return sum([s.__hash__() for s in str_list])


def remove_duplicate(para_list: List[ScoreParagraph]) -> Iterable[ScoreParagraph]:
    def para_hash(para : ScoreParagraph) -> int:
        return tokenlist_hash(para.paragraph.tokens)

    hash_list = lmap(para_hash, para_list)
    hash_set = set(hash_list)

    if len(hash_set) == len(hash_list):
        return para_list

    prev_hash = set()
    for i, hash_val in enumerate(hash_list):
        if hash_val in prev_hash:
            continue
        else:
            yield para_list[i]
        prev_hash.add(hash_val)


def extract_n_gram(max_para, para_claim_feature: ParagraphClaimPersFeature) -> PCNGramFeature:
    n_list = [1, 2, 3]
    all_n_grams = {n: Counter() for n in n_list}
    para_list = para_claim_feature.feature[:max_para]
    for para in remove_duplicate(para_list):
        for n in n_list:
            ngram_list = nltk.ngrams(para.paragraph.tokens, n)
            all_n_grams[n].update(ngram_list)

    return PCNGramFeature(claim_pers=para_claim_feature.claim_pers,
                          n_grams=all_n_grams)


class PCNgramWorker:
    def __init__(self, max_para, input_job_name, out_dir):
        self.out_dir = out_dir
        exist_or_mkdir(out_dir)
        self.input_dir = os.path.join(sydney_working_dir, input_job_name)
        self.max_para = max_para

    def work(self, job_id):
        features: List[ParagraphClaimPersFeature] = pickle.load(open(os.path.join(self.input_dir, str(job_id)), "rb"))
        all_result = []
        for f in features:
            r: PCNGramFeature = extract_n_gram(self.max_para, f)
            all_result.append(r)

        pickle.dump(all_result, open(os.path.join(self.out_dir, str(job_id)), "wb"))


