import random
from typing import List, Dict, Tuple, NamedTuple

from arg.perspectives.claim_lm.passage_common import score_over_zero
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from list_lib import lfilter, left, lfilter_not, lmap
from misc_lib import DataIDManager


class Instance(NamedTuple):
    candidate_text: str
    passage: List[str]
    is_correct: int
    data_id: int


class Generator(CPPNCGeneratorInterface):
    def __init__(self,
                 cid_to_passages: Dict[int, List[Tuple[List[str], float]]],
                 ):
        self.cid_to_passages = cid_to_passages

        self.all_cids = list(cid_to_passages.keys())

    def random_sample(self, exclude_cid) -> List[str]:
        cid = random.choice(self.all_cids)
        while cid == exclude_cid:
            cid = random.choice(self.all_cids)
        p, score = random.choice(self.cid_to_passages[cid])
        return p

    def generate_instances(self, claim: Dict, data_id_manager: DataIDManager) -> List[Instance]:
        cid = claim['cId']
        claim = claim['text']

        passages = self.cid_to_passages[cid]
        good_passages: List[List[str]] = left(lfilter(score_over_zero, passages))
        not_good_passages: List[List[str]] = left(lfilter_not(score_over_zero, passages))

        n_good = len(good_passages)
        n_not_good = len(not_good_passages)
        random_passage = list([self.random_sample(cid) for _ in range(10)])

        # len(pair_list_g_ng) = n_not_good   ( assuming n_not_good > n_good)

        def make_instance(passage, label):
            info = {
                'cid': cid
            }
            return Instance(claim,
                            passage,
                            label,
                            data_id_manager.assign(info))

        l1 = lmap(lambda p: make_instance(p, 1), good_passages)
        l2 = lmap(lambda p: make_instance(p, 0), not_good_passages)
        l3 = lmap(lambda p: make_instance(p, 0), random_passage)
        print("g: ng : rand = {} : {} : {}".format(len(l1), len(l2), len(l3)))
        return l1 + l2 + l3
