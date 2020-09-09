import random
from collections import OrderedDict
from typing import NamedTuple, List, Dict, Tuple

from arg.perspectives.claim_lm.passage_common import score_over_zero
from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import get_claim_perspective_id_dict
from arg.perspectives.ppnc.ppnc_decl import CPPNCGeneratorInterface
from data_generator.common import get_tokenizer
from data_generator.create_feature import create_int_feature
from list_lib import flatten_map, left, lfilter, lfilter_not, lmap, foreach
from tf_util.record_writer_wrap import RecordWriterWrap
from tlm.data_gen.pairwise_common import generate_pairwise_combinations, combine_features


class PairedInstance(NamedTuple):
    passage_good: List[str]
    passage_worse: List[str]
    candidate_text: str
    strict_good: int
    strict_bad: int


class Generator(CPPNCGeneratorInterface):
    def __init__(self,
                 cid_to_passages: Dict[int, List[Tuple[List[str], float]]],
                 ):
        self.cid_to_passages = cid_to_passages

        self.all_cids = list(cid_to_passages.keys())
        self.gold = get_claim_perspective_id_dict()

    def random_sample(self, exclude_cid) -> List[str]:
        cid = random.choice(self.all_cids)
        while cid == exclude_cid:
            cid = random.choice(self.all_cids)
        p, score = random.choice(self.cid_to_passages[cid])
        return p

    def generate_instances(self, claim: Dict, data_id_manager) -> List[PairedInstance]:
        cid = claim['cId']
        perspective_clusters: List[List[int]] = self.gold[cid]

        passages = self.cid_to_passages[cid]
        gold_candidate_texts: List[str] = flatten_map(perspective_getter, perspective_clusters)


        good_passages: List[List[str]] = left(lfilter(score_over_zero, passages))
        not_good_passages: List[List[str]] = left(lfilter_not(score_over_zero, passages))

        # print("good/not_good passages : {}/{}".format(len(good_passages), len(not_good_passages)))

        # make good vs not_good pairs
        # about 100 items
        pair_list_g_ng: List[Tuple[List[str], List[str]]] = generate_pairwise_combinations(not_good_passages, good_passages, True)
        # make not_good vs random pairs
        # about 100 items
        pair_list_ng_rand: List[Tuple[List[str], List[str]]] = list([(inst, self.random_sample(cid)) for inst in not_good_passages])

        # generate (candiate_texts) X (two pair_list), while limit maximum to 5  * len(two pair_list) = 1000
        max_insts = 100 * 2 * 5

        def infinite_passage_iterator():
            while True:
                for pair in pair_list_g_ng:
                    strict_good = 1
                    strict_bad = 0
                    yield pair, strict_good, strict_bad
                for pair in pair_list_ng_rand:
                    strict_good = 0
                    strict_bad = 1
                    yield pair, strict_good, strict_bad

        itr = infinite_passage_iterator()
        all_passage_pair_len = len(pair_list_g_ng) + len(pair_list_ng_rand)
        n_passage_per_inst = int(max_insts / len(gold_candidate_texts)) + 1
        n_passage_per_inst = min(all_passage_pair_len, n_passage_per_inst)

        all_insts = []
        for candidate in gold_candidate_texts:
            for _ in range(n_passage_per_inst):
                passage_pair, strict_good, strict_bad = itr.__next__()
                passage_good, passage_worse = passage_pair
                insts = PairedInstance(passage_good, passage_worse, candidate, strict_good, strict_bad)
                all_insts.append(insts)
        return all_insts



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
        tokens1: List[str] = tokenizer.tokenize(inst.candidate_text)
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
    features: List[OrderedDict] = lmap(encode, records)
    foreach(writer.write_feature, features)
    writer.close()


