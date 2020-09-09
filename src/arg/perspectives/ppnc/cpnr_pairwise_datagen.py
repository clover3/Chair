import random
from collections import OrderedDict
from typing import List, Dict, Tuple, NamedTuple

from arg.perspectives.claim_lm.passage_common import score_over_zero
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from data_generator.create_feature import create_int_feature
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import lfilter, left, lfilter_not, lmap, foreach
from misc_lib import DataIDManager
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.pairwise_common import generate_pairwise_combinations, combine_features


class PairedInstance(NamedTuple):
    query_text: str
    passage_good: List[str]
    passage_worse: List[str]
    strict_good: int
    strict_bad: int
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

    def generate_instances(self, claim: Dict, data_id_manager: DataIDManager) -> List[PairedInstance]:
        cid = claim['cId']
        claim = claim['text']

        passages = self.cid_to_passages[cid]
        good_passages: List[List[str]] = left(lfilter(score_over_zero, passages))
        not_good_passages: List[List[str]] = left(lfilter_not(score_over_zero, passages))

        n_good = len(good_passages)
        n_not_good = len(not_good_passages)

        # len(pair_list_g_ng) = n_not_good   ( assuming n_not_good > n_good)
        pair_list_g_ng: List[Tuple[List[str], List[str]]] = generate_pairwise_combinations(not_good_passages, good_passages, True)
        # len(pair_list_g_rand) = n_good
        pair_list_g_rand: List[Tuple[List[str], List[str]]] = list(
            [(inst, self.random_sample(cid)) for inst in good_passages])
        # len(pair_list_g_rand) = n_not_good
        pair_list_ng_rand: List[Tuple[List[str], List[str]]] = list([(inst, self.random_sample(cid)) for inst in not_good_passages])

        def make_instance(passage_pair, strict_good, strict_bad):
            passage_good, passage_worse = passage_pair
            info = {
                'cid': cid
            }
            return PairedInstance(claim,
                                  passage_good,
                                  passage_worse,
                                  strict_good,
                                  strict_bad,
                                  data_id_manager.assign(info))

        l1 = lmap(lambda pair: make_instance(pair, 1, 0), pair_list_g_ng)
        l2 = lmap(lambda pair: make_instance(pair, 0, 1), pair_list_ng_rand)
        l3 = lmap(lambda pair: make_instance(pair, 1, 1), pair_list_g_rand)
        print("g-ng : ng-rank : g-rand = {} : {} : {}".format(len(l1), len(l2), len(l3)))
        return l1 + l2 + l3


def write_records(records: List[PairedInstance],
                  max_seq_length,
                  output_path):
    tokenizer = get_tokenizer()

    def tokenize_from_tokens(tokens: List[str]) -> List[str]:
        output = []
        for t in tokens:
            ts = tokenizer.tokenize(t)
            output.extend(ts)
        return output

    def encode(inst: PairedInstance) -> OrderedDict:
        tokens1: List[str] = tokenizer.tokenize(inst.query_text)
        max_seg2_len = max_seq_length - 3 - len(tokens1)

        def concat_tokens(raw_tokens: List[str]):
            tokens2 = tokenize_from_tokens(raw_tokens)[:max_seg2_len]
            tokens = ["[CLS]"] + tokens1 + ["[SEP]"] + tokens2 + ["[SEP]"]

            segment_ids = [0] * (len(tokens1) + 2) \
                          + [1] * (len(tokens2) + 1)
            tokens = tokens[:max_seq_length]
            segment_ids = segment_ids[:max_seq_length]
            return tokens, segment_ids

        out_tokens1, seg1 = concat_tokens(inst.passage_good)
        out_tokens2, seg2 = concat_tokens(inst.passage_worse)
        features = combine_features(out_tokens1, seg1, out_tokens2, seg2, tokenizer, max_seq_length)
        features['strict_good'] = create_int_feature([inst.strict_good])
        features['strict_bad'] = create_int_feature([inst.strict_bad])
        return features

    writer = RecordWriterWrap(output_path)
    features_list: List[OrderedDict] = lmap(encode, records)
    foreach(writer.write_feature, features_list)
    writer.close()


