import random
from typing import List, Callable

from bert_api.segmented_instance.seg_instance import SegmentedInstance
from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from list_lib import left
from misc_lib import get_second, NamedAverager
from tlm.qtype.partial_relevance.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.eval_metric.segment_modify_fn import get_drop_non_zero, DocModFuncB
from tlm.qtype.partial_relevance.loader import load_mmde_problem
from tlm.qtype.partial_relevance.runner.run_eval_old.run_partial_related_full_eval import get_mmd_client


def get_highlighted_text(tokenizer, drop_indices, n_seg, text2):
    all_words = [ids_to_text(tokenizer, text2.get_tokens_for_seg(i)) for i in range(n_seg)]
    for i in drop_indices:
        all_words[i] = "[{}]".format(all_words[i])
    text = " ".join(all_words)
    return text


class Searcher:
    def __init__(self, forward_fn, doc_modify_fn):
        self.doc_modify_fn = doc_modify_fn
        self.forward_fn = forward_fn
        self.tokenizer = get_tokenizer()

    def search(self, problem: RelatedEvalInstance):
        stop = False
        n_seg = problem.seg_instance.text2.get_seg_len()
        source_indices = list(range(n_seg))
        text1 = problem.seg_instance.text1
        text2 = problem.seg_instance.text2
        base_score = self.get_score(text1, text2)
        n_sample = 13
        current_best_score = 0
        current_best_indices = []
        n_iter_w_sample_size = 0
        na = NamedAverager()
        while not stop:
            drop_indices = random.sample(source_indices, n_sample)
            new_text = text2.get_dropped_text(drop_indices)
            new_score = self.get_score(text1, new_text)
            change = base_score - new_score
            for i in drop_indices:
                na[i].append(change)

            if change > current_best_score:
                current_best_indices = drop_indices
                text = get_highlighted_text(self.tokenizer, drop_indices, n_seg, text2)
                print("New best: {0:.2f} {1}".format(change, text))
                current_best_score = change

            n_iter_w_sample_size += 1
            if n_iter_w_sample_size > 2000:
                n_sample += 1
                print("Increase n_sample to ", n_sample)
                n_iter_w_sample_size = 0
                #   Run average as run

                idx_drop_list = list(na.get_average_dict().items())
                idx_drop_list.sort(key=get_second, reverse=True)
                drop_indices = left(idx_drop_list)[:10]
                text_to_print = get_highlighted_text(self.tokenizer, drop_indices, n_seg, text2)
                new_text = text2.get_dropped_text(drop_indices)
                new_score = self.get_score(text1, new_text)
                change = base_score - new_score
                print("On average this is best ({0:.2f})".format(change))
                print(text_to_print)

    def get_score(self, text1, text2):
        inst = SegmentedInstance(text1, text2)
        return self.forward_fn([inst])[0]


def run_search(dataset, model_interface="localhost"):
    problems: List[RelatedEvalInstance] = load_mmde_problem(dataset)
    problems = [p for p in problems if p.score >= 0.5]
    problems = problems[:1]
    doc_mod_fn: DocModFuncB = get_drop_non_zero()
    forward_fn: Callable[[List[SegmentedInstance]], List[float]] = get_mmd_client(model_interface)

    searcher = Searcher(forward_fn, doc_mod_fn)

    for p in problems:
        searcher.search(p)


def main():
    run_search("dev_word")


if __name__ == "__main__":
    main()