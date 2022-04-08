from typing import List

from data_generator.tokenizer_wo_tf import get_tokenizer, ids_to_text
from tlm.qtype.partial_relevance.bert_mask_interface.bert_masking_client import get_localhost_bert_mask_client
from contradiction.alignment.data_structure.related_eval_instance import RelatedEvalInstance
from tlm.qtype.partial_relevance.loader import load_mmde_problem


def exact_match_check():
    problems: List[RelatedEvalInstance] = load_mmde_problem("dev_sent")
    problems = problems
    tokenizer = get_tokenizer()
    n_relevant = 0
    n_not_found = 0
    for p in problems:
        inst = p.seg_instance
        content_tokens = inst.text1.get_tokens_for_seg(1)
        doc_tokens= inst.text2.tokens_ids

        content_tokens_sig = " ".join(map(str, content_tokens))
        doc_tokens_sig = " ".join(map(str, doc_tokens))

        if p.score > 0.5:
            n_relevant += 1
            if content_tokens_sig not in doc_tokens_sig:
                n_not_found += 1
                query = ids_to_text(tokenizer, inst.text1.tokens_ids)
                doc = ids_to_text(tokenizer, inst.text2.tokens_ids)
                print("Content span: " + ids_to_text(tokenizer, content_tokens))
                print("Query: " + query)
                print("Score: {0:.2f}".format(p.score))
                print("Document: ", doc)
                print("-----------------------------")

    print("{} of {} has non-exact match".format(n_not_found, n_relevant))


def main():
    problems: List[RelatedEvalInstance] = load_mmde_problem("dev_sent")
    problems = problems
    predictor = get_localhost_bert_mask_client()
    max_seq_length = 512
    tokenizer = get_tokenizer()

    seen_query = set()
    for p in problems:
        inst = p.seg_instance
        query = ids_to_text(tokenizer, inst.text1.tokens_ids)
        doc = ids_to_text(tokenizer, inst.text2.tokens_ids)
        if query not in seen_query:
            input("Enter to continue")
            seen_query.add(query)
        print("PID: ", p.problem_id)
        print("Query: " + query)
        print("Score: {0:.2f}".format(p.score))
        print("Document: ", doc)
        print("-----------------------------")


if __name__ == "__main__":
    main()
