from collections import Counter, defaultdict
from typing import Tuple, Iterator

import spacy

from cache import save_to_pickle, load_from_pickle
from dataset_specific.msmarco.enum_documents import enum_documents
from misc_lib import TimeEstimator
from tlm.qtype.analysis_fde.fde_module import FDEModuleEx
from tlm.qtype.wikify.doc_span_count import word_count_per_ft, print_log_odd_per_span
from tlm.qtype.wikify.runner_s.wikify_dev import get_fde_module_ex


def enum_interesting_docs() -> Iterator[Tuple[str, str]]:
    target_entity = ["EVENT", "FAC", "GPE", "LANGUAGE", "LAW", "LOC", "NORP", "ORG", "PERSON", "PRODUCT", "WORK_OF_ART"]
    nlp = spacy.load("en_core_web_sm")
    n_max_word = 300
    for doc in enum_documents():
        spacy_obj = nlp(doc.title)
        space_tokenized = doc.body.split()

        body_head = " ".join(space_tokenized[:n_max_word])
        valid_entity_list = []
        if spacy_obj.ents:
            for e in spacy_obj.ents:
                if e.label_ in target_entity and str(e).lower() in body_head.lower():
                    valid_entity_list.append(e)

        doc_text = doc.title + " " + body_head

        if valid_entity_list:
            e = valid_entity_list[0]
            yield str(e), doc_text


# Check which words are matched to each of func_spans.
# Method: if sentence deletion results in the change of func_span scores,
# than consider all words in the sentence as matched
def main():
    run_name = "qtype_2Y_v_train_120000"
    fde_module: FDEModuleEx = get_fde_module_ex(run_name)
    per_span_dict_acc = defaultdict(Counter)
    tf_acc = Counter()
    max_iter = 500
    n_iter = 0
    ticker = TimeEstimator(max_iter)
    for e, text in enum_interesting_docs():
        per_span_dict, tf = word_count_per_ft(fde_module, e, text)
        for key, counter in per_span_dict.items():
            per_span_dict_acc[key].update(counter)
        tf_acc.update(tf)

        n_iter += 1
        print(n_iter)
        ticker.tick()
        if n_iter >= max_iter:
            break

    save_to_pickle(per_span_dict_acc, "per_span_dict_acc")
    save_to_pickle(tf_acc, "tf_acc")
    print_log_odd_per_span(per_span_dict_acc, tf_acc)


def debug():
    run_name = "qtype_2Y_v_train_120000"
    fde_module: FDEModuleEx = get_fde_module_ex(run_name)
    per_span_dict, tf = word_count_per_ft(fde_module, "entity", "")


def show():
    per_span_dict_acc = load_from_pickle("per_span_dict_acc")
    tf_acc = load_from_pickle("tf_acc")
    print_log_odd_per_span(per_span_dict_acc, tf_acc)


if __name__ == "__main__":
    show()
