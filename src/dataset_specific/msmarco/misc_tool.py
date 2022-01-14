def get_qid_to_job_id(query_group):
    qid_to_job_id = {}
    for job_id, qids in enumerate(query_group):
        for qid in qids:
            qid_to_job_id[qid] = job_id
    return qid_to_job_id