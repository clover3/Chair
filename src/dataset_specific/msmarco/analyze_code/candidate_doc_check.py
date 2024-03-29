from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)

    not_found_list = []
    for job_id, q_group in enumerate(resource.query_group):
        for qid in q_group:
            if qid not in resource.candidate_doc_d:
                print(job_id, qid)
                not_found_list.append(qid)

    print("{} queries do not have candidate".format(len(not_found_list)))


