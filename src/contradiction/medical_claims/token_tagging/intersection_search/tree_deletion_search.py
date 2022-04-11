from typing import Callable, Dict, Iterator, NamedTuple
from typing import List, Iterable

import scipy.special
import spacy

from bert_api.segmented_instance.segmented_text import SegmentedText, seg_to_text
from contradiction.medical_claims.token_tagging.intersection_search.intersect_search_if import IntersectionSearchIF
from contradiction.medical_claims.token_tagging.intersection_search.text_align_helper import \
    align_tokens_segmented_text
from bert_api.task_clients.nli_interface.nli_interface import NLIPredictorSig, NLIInput
from data_generator.NLI.enlidef import NEUTRAL
from data_generator.tokenizer_wo_tf import get_tokenizer
from list_lib import get_max_idx
from misc_lib import pick1


def align_tokens_segmented_text_old(tokenizer, tokens: List[str], t: SegmentedText) -> List[int]:
    word_list: List[str] = [t.get_seg_text(tokenizer, i) for i in t.enum_seg_idx()]
    last_idx = -1
    index_mapping = []
    print(tokens)
    print(t.get_readable_rep(tokenizer))
    for t in tokens:
        tl = t.lower()

        next_idx = -1
        for j in range(last_idx+1, len(word_list)):
            if tl == word_list[j]:
                next_idx = j
                break

        if next_idx > last_idx + 2:
            print("WARNING ")
            last_idx = next_idx
        elif next_idx == -1:
            print("Not found:", tl)
        else:
            last_idx = next_idx

        index_mapping.append(next_idx)
    return index_mapping


def align_spacy_tokens_segmented_text(tokenizer, spacy_tokens: Iterable, t: SegmentedText) -> Dict[int, List[int]]:
    return align_tokens_segmented_text(tokenizer, [str(s) for s in spacy_tokens], t)


class Subsequence(NamedTuple):
    segmented_text: SegmentedText
    parent: SegmentedText
    parent_drop_indices: List[int]

    def get_seg_len(self):
        return self.segmented_text.get_seg_len()


def do_local_search(init: Subsequence, get_score: Callable[[SegmentedText], float]):
    stop = len(init.parent_drop_indices) == 0
    best_score: float = get_score(init.segmented_text)
    best_sub: Subsequence = init
    n_try_wo_update = 0
    while not stop:
        new_point: Subsequence = add_one_token(best_sub)
        new_score = get_score(new_point.segmented_text)

        if new_score > best_score:
            best_score = new_score
            best_sub: Subsequence = new_point
            n_try_wo_update = 0
        else:
            n_try_wo_update += 1

        if n_try_wo_update > 2 * len(best_sub.parent_drop_indices):
            stop = True
        if len(best_sub.parent_drop_indices) == 0:
            stop = True

    return best_sub



def add_one_token(subsequence: Subsequence) -> Subsequence:
    add_index = pick1(subsequence.parent_drop_indices)
    new_drop_indices = [i for i in subsequence.parent_drop_indices if i != add_index]
    new_segmented_text = subsequence.parent.get_dropped_text(new_drop_indices)
    new_subsequence = Subsequence(new_segmented_text, subsequence.parent, new_drop_indices)
    return new_subsequence


class TreeDeletionSearch(IntersectionSearchIF):
    def __init__(self, predict, verbose=False):
        self.predict: NLIPredictorSig = predict
        self.tokenizer = get_tokenizer()
        self.spacy_nlp = spacy.load("en_core_web_sm")
        self.verbose = verbose

    def predict_single(self, single: NLIInput) -> int:
        logits = self.predict([single])[0]
        return get_max_idx(logits)

    def find_intersection(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        # tprint("find_intersection()")
        # This only runs one-sided search, actual intersection can be longer than t2_p
        t2_p = self.find_longest_non_neutral_subset(t1, t2)
        t1_p = self.find_longest_non_neutral_subset(t2, t1)
        if t1_p.get_seg_len() > t2_p.get_seg_len():
            return t1_p
        else:
            return t2_p

    def find_longest_non_neutral_subset(self, t1: SegmentedText, t2: SegmentedText) -> Subsequence:
        # This only runs one-sided search, actual intersection can be longer than t2_p
        best_t2_p: Subsequence = None
        best_len = 0
        for t2_p in self.enum_entailed_subset(t1, t2):
            if t2_p.get_seg_len() > best_len:
                best_len = t2_p.get_seg_len()
                best_t2_p = t2_p

        def subsequence_score(segmented: SegmentedText):
            if self.is_entailed(t1, segmented):
                return segmented.get_seg_len()
            else:
                return 0
        self.print("find_longest_non_neutral_subset()")
        self.print("After enum: {} {}".format(best_len, seg_to_text(self.tokenizer, best_t2_p.segmented_text)))
        new_best_tp = do_local_search(best_t2_p, subsequence_score)
        self.print("After local search: {} {}".format(new_best_tp.get_seg_len(),
                                                      seg_to_text(self.tokenizer, new_best_tp.segmented_text)))
        return new_best_tp

    def print(self, s):
        if self.verbose:
            print("TDS: " + s)
        else:
            pass

    def enum_entailed_subset(self, t1: SegmentedText, t2: SegmentedText) -> Iterator[Subsequence]:
        # TODO: For each word of t2
        # t2_p = subtree(word)
        # t2_p = t2 - subtree(word)
        spacy_tokens = self.spacy_nlp(seg_to_text(self.tokenizer, t2))
        spacy_to_st_idx: Dict[int, List[int]] = align_spacy_tokens_segmented_text(self.tokenizer, spacy_tokens, t2)
        ch_idx_to_token_idx = {}
        for t_idx, t in enumerate(spacy_tokens):
            ch_idx_to_token_idx[t.idx] = t_idx

        # for token in spacy_tokens:
        #     print(token, [ch_idx_to_token_idx[child.idx] for child in token.subtree])

        def get_prob_for(t: Subsequence):
            logits = self.predict([NLIInput(t1, t.segmented_text)])[0]
            probs = scipy.special.softmax(logits)
            return probs

        def enum_all_slices():
            for i in range(len(spacy_tokens)):
                child_indices = [ch_idx_to_token_idx[child.idx] for child in spacy_tokens[i].subtree]
                st_indices: List[int] = get_child_indices_translated(child_indices, spacy_to_st_idx)
                yield Subsequence(t2.get_dropped_text(st_indices), t2, st_indices)
                drop_indices = t2.get_remaining_indices(st_indices)
                yield Subsequence(t2.get_dropped_text(drop_indices), t2, drop_indices)

        for seg in enum_all_slices():
            label = get_max_idx(get_prob_for(seg))
            if label != NEUTRAL:
                yield seg

    def is_entailed(self, t1, t2):
        logits = self.predict([NLIInput(t1, t2)])[0]
        probs = scipy.special.softmax(logits)
        return get_max_idx(probs) != NEUTRAL


def get_child_indices_translated(child_indices, spacy_to_st_idx: Dict[int, List[int]]) -> List[int]:
    t2_indices = []
    for j in child_indices:
        st_indices = spacy_to_st_idx[j]
        t2_indices.extend(st_indices)

    filling_indices = []
    for i in range(len(t2_indices)-1):
        e1 = t2_indices[i]
        e2 = t2_indices[i+1]
        for k in range(e1+1, e2):
            filling_indices.append(k)

    if filling_indices:
        raise ValueError()

    t2_indices.extend(filling_indices)
    t2_indices.sort()
    return t2_indices