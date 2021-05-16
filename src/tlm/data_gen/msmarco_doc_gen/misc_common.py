def get_pos_neg_doc_ids_for_qid(resource, qid):
    pos_doc_id_list = []
    neg_doc_id_list = []
    for pos_doc_id in resource.get_doc_for_query_d()[qid]:

        label = resource.get_label(qid, pos_doc_id)
        if label:
            pos_doc_id_list.append(pos_doc_id)
        else:
            neg_doc_id_list.append(pos_doc_id)
    return pos_doc_id_list, neg_doc_id_list