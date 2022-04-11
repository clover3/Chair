import random
from abc import ABC, abstractmethod
from typing import List

import nltk
import numpy as np
import scipy.special

from bert_api.segmented_instance.segmented_text import SegmentedText, seg_to_text
from bert_api.task_clients.nli_interface.nli_interface import NLIPredictorSig, NLIInput
from data_generator.NLI.enlidef import ENTAILMENT, nli_label_list, NEUTRAL, nli_probs_str
from data_generator.tokenizer_wo_tf import get_tokenizer, pretty_tokens
from list_lib import get_max_idx
from misc_lib import tprint, Averager


class IntersectionSearchIF(ABC):
    @abstractmethod
    def find_intersection(self, t1: SegmentedText, t2: SegmentedText):
        pass


def delete_random_word(s: SegmentedText) -> SegmentedText:
    drop_idx: int = random.randint(0, s.get_seg_len()-1)
    return s.get_dropped_text([drop_idx])


class WordDeletionSearch(IntersectionSearchIF):
    def __init__(self, predict):
        self.predict: NLIPredictorSig = predict
        self.tokenizer = get_tokenizer()

    def predict_single(self, single: NLIInput) -> int:
        logits = self.predict([single])[0]
        return get_max_idx(logits)

    def find_intersection(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        tprint("find_intersection()")
        # This only runs one-sided search, actual intersection can be longer than t2_p
        t2_p = self._find_intersection_inner(t1, t2)
        t1_p = self._find_intersection_inner(t2, t1)
        if t1_p.get_seg_len() > t2_p.get_seg_len():
            return t1_p
        else:
            return t2_p

    def _find_intersection_inner(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        tprint("find_intersection()")
        # This only runs one-sided search, actual intersection can be longer than t2_p
        best_t2_p: SegmentedText = None
        best_len = 0
        while True:
            t2_p = self.get_entailed_subset(t1, t2)
            decision = self.predict_single(NLIInput(t1, t2_p))
            if decision is NEUTRAL:
                print("Error")
            if t2_p.get_seg_len() > best_len:
                best_len = t2_p.get_seg_len()
                best_t2_p = t2_p
                print("Best : {} ({})".format(seg_to_text(self.tokenizer, best_t2_p), best_len))

        return best_t2_p

    def get_entailed_subset(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        # tprint("get_entailed_subset()")
        # tprint("prem: {}".format(seg_to_text(self.tokenizer, t1)))
        # delete words (segments) from t2, until the decision becomes entailment
        last_decision = self.predict_single(NLIInput(t1, t2))
        n_retry = 5
        t2_p = t2
        while last_decision == NEUTRAL and t2_p.get_seg_len() > 0:
            t2_p = delete_random_word(t2_p)
            decision = self.predict_single(NLIInput(t1, t2_p))
            decision_str = nli_label_list[decision]
            s = seg_to_text(self.tokenizer, t2_p)
            # tprint("{} segments, {} : {}".format(t2_p.get_seg_len(),
            #                                      decision_str,
            #                                      s))
            last_decision = decision
            if t2_p.get_seg_len() == 0 and n_retry > 0:
                n_retry -= 1
                t2_p = t2
                last_decision = NEUTRAL
        return t2_p


def sample_subseq_to_delete(segment: SegmentedText):
    n_seg = segment.get_seg_len()
    G_del_factor = 0.7

    def sample_len():
        l = 1
        v = random.random()
        while v < G_del_factor and l < n_seg:
            l = l * 2
            v = random.random()
        return min(l, n_seg)

    del_len = sample_len()
    start_idx = random.randint(0, n_seg - 1)
    end_idx = min(start_idx + del_len, n_seg - 1)
    drop_indices = list(range(start_idx, end_idx))
    return drop_indices


class SeqDeletionSearch(IntersectionSearchIF):
    def __init__(self, predict, signal_fn):
        self.predict: NLIPredictorSig = predict
        self.tokenizer = get_tokenizer()
        self.signal_fn = signal_fn

    def get_score(self, single: NLIInput) -> int:
        logits = self.predict([single])[0]
        probs = scipy.special.softmax(logits)
        return self.signal_fn(probs)

    def find_intersection(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        tprint("find_intersection()")
        # This only runs one-sided search, actual intersection can be longer than t2_p
        t2_p = self.get_entailed_subset(t1, t2)
        # t1_p = self.get_entailed_subset(t2, t1)
        return t2_p

    def get_entailed_subset(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        tprint("get_entailed_subset()")
        tprint("prem: {}".format(seg_to_text(self.tokenizer, t1)))
        tprint("hypo: {}".format(seg_to_text(self.tokenizer, t2)))

        base_score = self.get_score(NLIInput(t1, t2))
        n_seg = t2.get_seg_len()
        avg_scores: List[Averager] = [Averager() for _ in range(n_seg)]

        for _ in range(30):
            drop_indices = sample_subseq_to_delete(t2)
            t2_p = t2.get_dropped_text(drop_indices)
            n_score = self.get_score(NLIInput(t1, t2_p))
            diff = base_score - n_score
            for i in drop_indices:
                avg_scores[i].append(diff)
            s = seg_to_text(self.tokenizer, t2_p)
            print("{}: {}".format(n_score, s))

        out_s = []
        for i in t2.enum_seg_idx():
            word = pretty_tokens(self.tokenizer.convert_ids_to_tokens(t2.get_tokens_for_seg(i)), True)
            s = "[{0} {1:.2f}]".format(word, avg_scores[i].get_average())
            out_s.append(s)
        print(" ".join(out_s))


class PhraseSearch(IntersectionSearchIF):
    def __init__(self, predict):
        self.predict: NLIPredictorSig = predict
        self.tokenizer = get_tokenizer()

    def get_probs(self, single: NLIInput) -> List[float]:
        logits = self.predict([single])[0]
        probs = scipy.special.softmax(logits)
        return probs

    def find_intersection(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        tprint("find_intersection()")
        # This only runs one-sided search, actual intersection can be longer than t2_p
        t2_p = self.get_entailed_subset(t1, t2)
        t1_p = self.get_entailed_subset(t2, t1)
        # t1_p = self.get_entailed_subset(t2, t1)
        return t2_p

    def get_entailed_subset(self, t1: SegmentedText, t2: SegmentedText) -> SegmentedText:
        tprint("get_entailed_subset()")
        tprint("prem: {}".format(seg_to_text(self.tokenizer, t1)))
        tprint("hypo: {}".format(seg_to_text(self.tokenizer, t2)))

        n_seg = t2.get_seg_len()
        # TODO : for n-gram in t2, test if entail(t1, n_gram_i)
        source_indices = list(range(n_seg))
        prob_per_locations: List[List[List[float]]] = [list() for _ in range(n_seg)]
        for n in [1, 2, 3]:
            indices_list = nltk.ngrams(source_indices, n)
            for indices in indices_list:
                t2_p = t2.get_sliced_text(indices)
                probs = self.get_probs(NLIInput(t1, t2_p))
                for j in indices:
                    prob_per_locations[j].append(probs)

        one_gram_probs = [l[0] for l in prob_per_locations]

        print("1-gram")

        def get_one_gram(i):
            return nli_probs_str(one_gram_probs[i])

        print(self.build_text_summary(get_one_gram, t2))

        def get_avg(i):
            avg_probs = np.mean(np.array(prob_per_locations[i]), axis=0)
            return nli_probs_str(avg_probs)

        print("avg")
        print(self.build_text_summary(get_avg, t2))
        common_indices = []
        for i in t2.enum_seg_idx():
            avg_probs = np.mean(np.array(prob_per_locations[i]), axis=0)
            if get_max_idx(avg_probs) == ENTAILMENT:
                common_indices.append(i)

        return t2.get_sliced_text(common_indices)

    def build_text_summary(self, get_score_str_fn, t2):
        s_list = []
        for i in t2.enum_seg_idx():
            s1 = t2.get_seg_text(self.tokenizer, i)
            s2 = get_score_str_fn(i)
            s_list.append("[{} {}]".format(s1, s2))
        s_out = " ".join(s_list)
        return s_out


