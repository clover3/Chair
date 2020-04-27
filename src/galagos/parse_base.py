def write_ranked_list_from_d(ranked_list_dict, out_path):
    f = open(out_path, "w")
    for q_id, ranked_list in ranked_list_dict.items():
        for doc_id, rank, score in ranked_list:
            line = "{} Q0 {} {} {} galago\n".format(q_id, doc_id, rank, score)
            f.write(line)
    f.close()


