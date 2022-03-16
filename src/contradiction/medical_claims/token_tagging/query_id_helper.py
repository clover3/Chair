

def get_query_id(group_no, inner_idx, sent_name, tag_type):
    query_id = "{}_{}_{}_{}".format(group_no, inner_idx, sent_name, tag_type)
    return query_id