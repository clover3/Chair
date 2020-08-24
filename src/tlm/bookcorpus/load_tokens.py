import pickle


def load_doc_seg(doc_id):
    if doc_id < 40:
        file_path = "/mnt/nfs/work3/youngwookim/data/bert_tf/bookcorpus_tokens/1_{}".format(doc_id)
    else:
        doc_id = doc_id - 40
        file_path = "/mnt/nfs/work3/youngwookim/data/bert_tf/bookcorpus_tokens2/2_{}".format(doc_id)

    f = open(file_path, "rb")
    return pickle.load(f)


def load_seg_with_repeat(job_id):
    doc_id = job_id % 75
    return load_doc_seg(doc_id)