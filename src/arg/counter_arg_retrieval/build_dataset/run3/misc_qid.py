from arg.counter_arg_retrieval.build_dataset.ca_query import CAQuery
from arg.perspectives.load import get_perspective_dict


class CA3QueryIDGen:
    def __init__(self):
        d = get_perspective_dict()
        p_text_to_id = {v: k for k, v in d.items()}
        self.p_text_to_id = p_text_to_id

    def get_qid(self, query: CAQuery):
        pid = self.p_text_to_id[query.perspective]
        qid = "{}_{}".format(query.qid, pid)
        return qid