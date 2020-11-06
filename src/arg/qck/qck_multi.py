import collections
from collections import OrderedDict
from typing import List, Iterable, Dict, Tuple, NamedTuple

from arg.qck.decl import QCKQuery, KDP, QKUnit, QCKInstance, QCKCandidate, KnowledgeDocumentPart
from arg.qck.qck_worker import InstanceGenerator
from data_generator.tokenizer_wo_tf import get_tokenizer, tokenize_from_tokens
from list_lib import flatten, lmap
from misc_lib import DataIDManager
from tlm.data_gen.bert_data_gen import create_int_feature


class Instance(NamedTuple):
    query_text: str
    candidate_text: str
    doc_tokens_list: List[List[str]]
    data_id: int
    is_correct: int


class InstanceTokenized(NamedTuple):
    passage_subtokens_list: List[List[str]]
    q_tokens: List[str]
    c_tokens: List[str]
    data_id: int
    is_correct: int


def drop_tokens(passages: List[KDP]):
    new_p_list = []
    for p in passages:
        new_p = KnowledgeDocumentPart(p.doc_id, p.passage_idx, p.start_location, [])
        new_p_list.append(new_p)
    return new_p_list


class MultiDocInstanceGenerator(InstanceGenerator):
    def __init__(self,
                 candidates_dict: Dict[str, List[QCKCandidate]],
                 is_correct_fn,
                 config,
                 ):
        self.sent_max_length = config['sent_max_length']
        self.doc_max_length = config['doc_max_length']
        self.num_docs = config['num_docs']
        self.tokenizer = get_tokenizer()
        self.candidates_dict: Dict[str, List[QCKCandidate]] = candidates_dict
        self._is_correct = is_correct_fn

    def generate(self,
                 kc_candidate: Iterable[QKUnit],
                 data_id_manager: DataIDManager) -> Iterable[QCKInstance]:

        def convert(pair: Tuple[QCKQuery, List[KDP]]) -> Iterable[QCKInstance]:
            query, passages = pair
            candidates = self.candidates_dict[query.query_id]
            for c in candidates:

                idx = 0
                while idx < len(passages):
                    sel_passages: List[KDP] = passages[idx:idx+self.num_docs]
                    idx += self.num_docs
                    info = {
                                'query': query,
                                'candidate': c,
                                'kdp_list': drop_tokens(sel_passages)
                            }
                    yield Instance(
                        query.text,
                        c.text,
                        list([p.tokens for p in sel_passages]),
                        data_id_manager.assign(info),
                        self._is_correct(query, c)
                    )

        return flatten(lmap(convert, kc_candidate))

    def _convert_sub_token(self, r: Instance) -> InstanceTokenized:
        tokenizer = self.tokenizer
        passage_subtokens_list = list([tokenize_from_tokens(tokenizer, p) for p in r.doc_tokens_list])
        tokens1: List[str] = tokenizer.tokenize(r.query_text)
        tokens2: List[str] = tokenizer.tokenize(r.candidate_text)

        return InstanceTokenized(passage_subtokens_list=passage_subtokens_list,
                               q_tokens=tokens1,
                               c_tokens=tokens2,
                               data_id=r.data_id,
                               is_correct=r.is_correct,
                               )

    def encode_fn(self, inst_raw: Instance) -> OrderedDict:
        inst: InstanceTokenized = self._convert_sub_token(inst_raw)
        q_tokens: List[str] = inst.q_tokens
        c_tokens: List[str] = inst.c_tokens
        passage_tokens_list: List[List[str]] = inst.passage_subtokens_list

        def enc(tokens, max_length):
            tokens = ["[CLS]"] + tokens
            if len(tokens) > max_length:
                print(len(tokens), end=" ")
                tokens = tokens[:max_length]

            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            input_masks = [1] * len(input_ids)
            while len(input_ids) < max_length:
                input_ids.append(0)
                input_masks.append(0)

            return input_ids, input_masks

        q_input_ids, q_input_masks = enc(q_tokens, self.sent_max_length)
        c_input_ids, c_input_masks = enc(c_tokens, self.sent_max_length)

        d_input_ids_all = []
        d_input_masks_all = []
        for p in passage_tokens_list:
            d_input_ids, d_input_masks = enc(p, self.doc_max_length)
            d_input_ids_all.extend(d_input_ids)
            d_input_masks_all.extend(d_input_masks)

        d_expected_len = self.num_docs * self.doc_max_length
        if len(d_input_ids_all) < d_expected_len:
            add_len = d_expected_len - len(d_input_ids_all)
            d_input_ids_all.extend([0] * add_len)
            d_input_masks_all.extend([0] * add_len)

        features = collections.OrderedDict()
        features['q_input_ids'] = create_int_feature(q_input_ids)
        features['q_input_masks'] = create_int_feature(q_input_masks)
        features['c_input_ids'] = create_int_feature(c_input_ids)
        features['c_input_masks'] = create_int_feature(c_input_masks)
        features['d_input_ids'] = create_int_feature(d_input_ids_all)
        features['d_input_masks'] = create_int_feature(d_input_masks_all)

        features['label_ids'] = create_int_feature([inst.is_correct])
        features['data_id'] = create_int_feature([inst.data_id])
        return features

