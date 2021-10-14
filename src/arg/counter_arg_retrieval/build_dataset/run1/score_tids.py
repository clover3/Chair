from typing import List, Iterable

from arg.counter_arg_retrieval.build_dataset.run1.load_resource import get_run1_resource
from arg.counter_arg_retrieval.scorer.tids_aawd import get_aawd_tids_svm
from bert_api.doc_score_helper import get_cache_doc_tokenizer
from data_generator.tokenize_helper import TokenizedText
from list_lib import get_max_idx
from misc_lib import enum_segments


def main():
    rlg, topics_v2, docs_d = get_run1_resource()
    get_tokenized_doc = get_cache_doc_tokenizer(docs_d)
    window_size = 25
    tids_scorer = get_aawd_tids_svm()

    #  For each (document) ->
    #    doc level score score
    #    passage level score
    #    sent level score
    def get_segments(doc) -> Iterable[str]:
        for st_sb, ed_sb in enum_segments(doc.sbword_tokens, window_size):
            st = doc.sbword_mapping[st_sb]
            ed = doc.sbword_mapping[ed_sb] if ed_sb < len(doc.sbword_mapping) else doc.sbword_mapping[-1] + 1
            tokens = doc.tokens[st:ed]
            yield " ".join(tokens)

    for topic_idx, topic in enumerate(topics_v2):
        print("Topic {}".format(topic_idx))
        print(topic.claim_text)
        print(topic.target_p[0])
        for doc_e in rlg[topic.ca_cid][:10]:
            doc: TokenizedText = get_tokenized_doc(doc_e.doc_id)
            segs: List[str] = list(get_segments(doc))
            if segs:
                score_list = tids_scorer.predict(segs)
                max_score = max(score_list)
                max_idx = get_max_idx(score_list)
                if max_score > 0:
                    print(doc_e.doc_id, segs[max_idx])



if __name__ == "__main__":
    main()