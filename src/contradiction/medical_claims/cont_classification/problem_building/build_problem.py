import itertools
import json
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Iterator

from cache import named_tuple_to_json, save_list_to_jsonl_w_fn
from contradiction.medical_claims.annotation_1.load_data import load_reviews_for_split
from contradiction.medical_claims.cont_classification.defs import ContProblem, NOTE_POS_PAIR, NOTE_NEG_TYPE1_YS, \
    NOTE_NEG_TYPE1_NO, NOTE_NEG_TYPE2
from contradiction.medical_claims.cont_classification.path_helper import get_problem_path, get_problem_note_path
from contradiction.medical_claims.load_corpus import Review, Claim
from contradiction.medical_claims.pilot.pilot_annotation import enum_true_instance
from trainer_v2.per_project.tli.qa_scorer.bm25_system import BM25TextPairScorer
from trainer_v2.per_project.tli.bioclaim_qa.eval_helper import get_bioclaim_retrieval_corpus
from list_lib import get_max_idx, right


def get_doc_id_table(docs) -> Tuple[Dict, Dict]:
    claim_to_doc_id = {}
    doc_id_to_claim = {}
    for doc_id, claim in docs:
        claim_to_doc_id[claim] = doc_id
        doc_id_to_claim[doc_id] = claim
    return claim_to_doc_id, doc_id_to_claim


# Claim is a data structure.
def separate_yes_no(claim_list: List[Claim]):
    yes_c = []
    no_c = []
    for c in claim_list:
        if c.assertion == "YS":
            yes_c.append(c)
        elif c.assertion == "NO":
            no_c.append(c)
        else:
            assert False
    return yes_c, no_c


def build_problem_set(split, scorer) -> Tuple[List[ContProblem], Dict]:
    review_list: List[Tuple[int, Review]] = load_reviews_for_split(split)

    queries, docs = get_bioclaim_retrieval_corpus(split)
    claim_text_to_doc_id, doc_id_to_claim_text = get_doc_id_table(docs)

    doc_id_to_group_no = defaultdict(list)
    for group_no, review in review_list:
        for c in review.claim_list:
            doc_id = claim_text_to_doc_id[c.text]
            doc_id_to_group_no[doc_id].append(group_no)

    claim_pair_note = {}

    n_per_topic = 5
    problems = []
    for review, pairs in enum_true_instance():
        random.shuffle(pairs)
        pairs = pairs[:n_per_topic]
        for c1, c2 in pairs:
            p = ContProblem(c1.question, c1.text, c2.text, 1)
            claim_pair_note[p.signature()] = NOTE_POS_PAIR
            problems.append(p)
    # True YES/NO Pair

    def add_problems(pair_itr, note_msg, max_n_per_topic):
        n_added = 0
        for c1, c2 in pair_itr:
            p = ContProblem(c1.question, c1.text, c2.text, 0)
            claim_pair_note[p.signature()] = note_msg
            problems.append(p)
            n_added += 1
            if n_added >= max_n_per_topic:
                break

    for group_no, review in review_list:
        yes_claims, no_claims = separate_yes_no(review.claim_list)
        random.shuffle(yes_claims)
        pair_itr = itertools.combinations(yes_claims, 2)
        add_problems(pair_itr, NOTE_NEG_TYPE1_YS, 5)

        random.shuffle(no_claims)
        pair_itr = itertools.combinations(no_claims, 2)
        add_problems(pair_itr, NOTE_NEG_TYPE1_NO, 5)

    # For each question, select YES/NO by high similarity
    def iter_claims_that_are_not_group_no(exclude_group_no):
        for group_no, review in review_list:
            if group_no != exclude_group_no:
                yield from review.claim_list

    for group_no, review in review_list:
        def get_pair_itr():
            claims = list(review.claim_list)
            claim_text_set = set([c.text for c in claims])
            random.shuffle(claims)
            c2_itr: Iterator[Claim] = iter_claims_that_are_not_group_no(group_no)
            c2_itr = [c for c in c2_itr if c.text not in claim_text_set]
            for c1 in claims:
                c2_opposite_itr = [c2 for c2 in c2_itr if c2.assertion != c1.assertion]
                if not c2_opposite_itr:
                    continue
                scores = [scorer(c1.text, c2.text) for c2 in c2_opposite_itr]
                max_idx = get_max_idx(scores)
                c2 = c2_opposite_itr[max_idx]
                yield c1, c2

        add_problems(get_pair_itr(), NOTE_NEG_TYPE2, 5)

    pos_problems = [p for p in problems if p.label]
    neg_problems = [p for p in problems if not p.label]
    print("{} pos_problems".format(len(pos_problems)))
    print("{} neg_problems".format(len(neg_problems)))

    indices = list(range(len(neg_problems)))
    random.shuffle(indices)
    sel_indices = indices[:len(pos_problems)]
    neg_problems = [p for i, p in enumerate(neg_problems) if i in sel_indices]
    problems = pos_problems + neg_problems
    note_hash = [p.signature() for p in problems]
    claim_pair_note = {k: v for k, v in claim_pair_note.items() if k in note_hash}

    return problems, claim_pair_note


def main():
    for split in ["dev", "test"]:
        _, claims = get_bioclaim_retrieval_corpus(split)
        system = BM25TextPairScorer(right(claims))
        problems, claim_pair_note = build_problem_set(split, system.score)
        save_list_to_jsonl_w_fn(problems, get_problem_path(split), named_tuple_to_json)
        json.dump(claim_pair_note, open(get_problem_note_path(split), "w"))


if __name__ == "__main__":
    main()