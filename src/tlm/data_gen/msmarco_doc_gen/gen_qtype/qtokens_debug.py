from dataset_specific.msmarco.common import load_candidate_doc_list_10
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourceTitleBodyTokensListTrain
from tlm.qtype.qid_to_content_tokens import get_qid_to_content_tokens


if __name__ == "__main__":
    split = "train"
    resource = ProcessedResourceTitleBodyTokensListTrain(split, load_candidate_doc_list_10)
    qid_to_entity_tokens = get_qid_to_content_tokens(split)

    for qids in resource.query_group:
        print(len(qids))
        for qid in qids:
            q_tokens = resource.get_q_tokens(qid)
            try:
                entity_tokens = qid_to_entity_tokens[qid]

                if not len(entity_tokens) <= len(q_tokens):
                    print("{} > {}".format(len(entity_tokens), len(q_tokens)))
                    print(entity_tokens)
                    print(q_tokens)
            except KeyError:
                print(qid, "not found")
