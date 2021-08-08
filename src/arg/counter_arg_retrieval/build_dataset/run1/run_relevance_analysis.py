import os
import pickle
from typing import List, Tuple

from tqdm import tqdm

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopicv2
from arg.counter_arg_retrieval.build_dataset.run1.load_resource import get_run1_resource
from arg.counter_arg_retrieval.scorer.relevance_analysis import AnalyzedDocument, analyze_doc_wrt_ca_topic
from bert_api.doc_score_helper import DocumentScorer, get_cache_doc_tokenizer
from bert_api.doc_score_helper import TokenizedText
from bert_api.msmarco_rerank import get_msmarco_client
from cache import save_to_pickle
from cpath import output_path


def save_rel(topic, ad_list):
    dir_path = os.path.join(output_path, "ca_building", "run1", "relevance_analysis")
    pickle.dump(ad_list, open(os.path.join(dir_path, topic.ca_cid), "wb"))


# For each document, score
# - Relevance to the claim
# - Relevance to the perspectives
def main():
    rlg, topics_v2, docs_d = get_run1_resource()

    client = get_msmarco_client()
    document_scorer = DocumentScorer(client, 20)

    print("After filtering")

    get_tokenized_doc = get_cache_doc_tokenizer(docs_d)
    save_payload: List[Tuple[CaTopicv2, List[AnalyzedDocument]]] = []

    ticker = tqdm(total=len(topics_v2) * 10)
    for topic_idx, topic in enumerate(topics_v2):
        ad_list = []
        print("Topic {}".format(topic_idx))
        for doc_e in rlg[topic.ca_cid][:10]:
            ticker.update(1)
            doc_payload: TokenizedText = get_tokenized_doc(doc_e.doc_id)
            ad: AnalyzedDocument = analyze_doc_wrt_ca_topic(document_scorer, topic, doc_payload)
            ad_list.append(ad)
        save_rel(topic, ad_list)
        save_payload.append((topic, ad_list))

    save_to_pickle(save_payload, "run_relevance_analysis")


def client_test():
    client = get_msmarco_client()
    client.send_payload([])


if __name__ == "__main__":
    main()
