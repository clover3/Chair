from data_generator.data_parser.robust2 import load_2k_rank, load_robust_qrel


def compare():
    qrel = load_robust_qrel()
    ranked_list = load_2k_rank()

    for q_id in qrel:
        galago_ranked_2k = [x[0] for x in ranked_list[q_id]][:100]
        docs_from_qrel = list(qrel[q_id].keys())

        print(len(galago_ranked_2k))
        print(len(docs_from_qrel))
        for doc_id in galago_ranked_2k:
            if doc_id not in docs_from_qrel:
                print(doc_id)



compare()