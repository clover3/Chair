from typing import List, Callable, Iterable, Dict, Tuple, NamedTuple
import scipy.special

from bert_api.segmented_instance.segmented_text import SegmentedText, get_word_level_segmented_text_from_str
from cache import save_list_to_jsonl_w_fn, load_list_from_jsonl
from datastore.sql_based_cache_client import SQLBasedCacheClientS


class NLIInput(NamedTuple):
    prem: SegmentedText
    hypo: SegmentedText

    def str_hash(self):
        return str(self.prem.tokens_ids) + str(self.hypo.tokens_ids)

    def to_json(self):
        return {'prem': self.prem.to_json(),
                'hypo': self.hypo.to_json()}

    @classmethod
    def from_json(cls, j):
        prem = SegmentedText.from_json(j['prem'])
        hypo = SegmentedText.from_json(j['hypo'])
        return NLIInput(prem, hypo)


def save_nli_inputs_to_jsonl(save_path, items: List[NLIInput]):
    def to_json(nli_input: NLIInput):
        return {
            'hash': nli_input.str_hash(),
            'content': nli_input.to_json()
        }

    save_list_to_jsonl_w_fn(items, save_path, to_json)


def load_nli_input_from_jsonl(save_path) -> List[NLIInput]:
    def from_json(j) -> NLIInput:
        return NLIInput.from_json(j['content'])

    return load_list_from_jsonl(save_path, from_json)


NLIPredictorFromSegTextSig = Callable[[List[NLIInput]], List[List[float]]]


def predict_from_text_pair(client: SQLBasedCacheClientS, tokenizer, text1, text2):
    t_text1: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, text1)
    t_text2: SegmentedText = get_word_level_segmented_text_from_str(tokenizer, text2)

    logits = client.predict([NLIInput(t_text1, t_text2)])[0]
    probs = scipy.special.softmax(logits)
    return probs
