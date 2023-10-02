import random


class PosNegSampler:
    def __init__(self, qrel):
        self.qrel = qrel

    def split_pos_neg_entries(self, entries, qid=None):
        if qid is None:
            qid = entries[0][0]

        pos_doc_ids = []
        for doc_id, score in self.qrel[qid].items():
            if score > 0:
                pos_doc_ids.append(doc_id)

        pos_doc = []
        neg_doc = []
        for e in entries:
            qid, pid, query, text = e
            if pid in pos_doc_ids:
                pos_doc.append(e)
            else:
                neg_doc.append(e)
        return pos_doc, neg_doc

    def sample_pos_neg(self, group):
        pos_docs, neg_docs = self.split_pos_neg_entries(group)
        if len(neg_docs) == 0:
            raise IndexError
        neg_idx = random.randrange(len(neg_docs))
        return pos_docs[0], neg_docs[neg_idx]
