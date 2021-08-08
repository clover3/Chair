import os
import pickle
from typing import List, Dict, Tuple

from tqdm import tqdm

from arg.counter_arg_retrieval.build_dataset.ca_types import CaTopic, CaTopicv2, get_ca2_converter
from arg.counter_arg_retrieval.build_dataset.resources import ca_building_q_res_path, \
    load_step2_claims_as_ca_topic, load_run1_doc_indexed
from arg.counter_arg_retrieval.scorer.relevance_analysis import AnalyzedDocument, analyze_doc_wrt_ca_topic
from bert_api.doc_score_helper import DocumentScorer
from bert_api.doc_score_helper import TokenizedText
from bert_api.msmarco_rerank import get_msmarco_client
from cache import save_to_pickle
from cpath import output_path
from list_lib import lfilter, lmap
from trec.trec_parse import load_ranked_list_grouped


def save_rel(topic, ad_list):
    dir_path = os.path.join(output_path, "ca_building", "run1", "relevance_analysis")
    pickle.dump(ad_list, open(os.path.join(dir_path, topic.ca_cid), "wb"))


# For each document, score
# - Relevance to the claim
# - Relevance to the perspectives
def main():
    topics: List[CaTopic] = load_step2_claims_as_ca_topic()
    rlg = load_ranked_list_grouped(ca_building_q_res_path)
    docs_d: Dict[str, str] = load_run1_doc_indexed()
    topics: List[CaTopic] = lfilter(lambda topic: topic.ca_cid in rlg, topics)
    topics_v2: List[CaTopicv2] = lmap(get_ca2_converter(), topics)

    client = get_msmarco_client()
    document_scorer = DocumentScorer(client, 20)

    print("After filtering")
    doc_payload_d = {}

    def get_tokenized_doc(doc_id) -> TokenizedText:
        if doc_id in doc_payload_d:
            return doc_payload_d[doc_id]
        return TokenizedText.from_text(docs_d[doc_id])

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
