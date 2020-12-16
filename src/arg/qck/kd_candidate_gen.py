from typing import List, Iterable, Dict, Tuple

from arg.perspectives.clueweb_db import load_doc
from arg.qck.decl import QCKQuery, KnowledgeDocument, KnowledgeDocumentPart, KDP
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list
from galagos.types import GalagoDocRankEntry
# KD = Knowledge Document
from misc_lib import TimeEstimator, tprint


def preload_docs(ranked_list: Dict[str, List[GalagoDocRankEntry]], top_n):
    all_doc_ids = set()
    for entries in ranked_list.values():
        for entry in entries[:top_n]:
            all_doc_ids.add(entry.doc_id)

    tprint(f"total of {len(all_doc_ids)} docs")
    tprint("Accessing DB")
    #  Get the doc from DB

    doc_ids_list = list(all_doc_ids)
    block_size = 1000
    idx = 0
    while idx < len(doc_ids_list):
        print(idx)
        st = idx
        ed = idx + block_size
        preload_man.preload(TokenizedCluewebDoc, doc_ids_list[st:ed])
        idx += block_size

    #preload_man.preload(TokenizedCluewebDoc, all_doc_ids)
    tprint("Done")


def get_duplicate(doc_list: List[KnowledgeDocument]):
    def para_hash(doc: KnowledgeDocument):
        return " ".join(doc.tokens)

    hash_set = set()
    duplicates = []
    for doc in doc_list:
        hash = para_hash(doc)
        if hash in hash_set:
            duplicates.append(doc.doc_id)
            continue

        hash_set.add(hash)
    return duplicates


def iterate_docs(q_res: List[GalagoDocRankEntry], top_n: int) -> Iterable[KnowledgeDocument]:
    docs = []
    for i in range(top_n):
        try:
            tokens = load_doc(q_res[i].doc_id)
            kd = KnowledgeDocument(q_res[i].doc_id, tokens)
            docs.append(kd)
        except KeyError:
            pass

    if len(docs) < top_n:
        print("Retrieved {} of {} docs".format(len(docs), top_n))

    duplicate_doc_ids = get_duplicate(docs)
    unique_docs = [d for d in docs if d.doc_id not in duplicate_doc_ids]
    return unique_docs


def iterate_document_parts(docs: Iterable[KnowledgeDocument], window_size, step_size, max_per_doc=999) -> List[KnowledgeDocumentPart]:
    # knowledge document parts list
    kdp_list: List[KnowledgeDocumentPart] = []
    for doc in docs:
        idx = 0
        passage_idx = 0
        while idx < len(doc.tokens) and passage_idx < max_per_doc:
            tokens = doc.tokens[idx:idx + window_size]
            kdp = KnowledgeDocumentPart(doc.doc_id, passage_idx, idx, tokens)
            kdp_list.append(kdp)
            idx += step_size
            passage_idx += 1
    return kdp_list


def qk_candidate_gen(q_res_path: str, queries: List[QCKQuery], top_n, config) -> List[Tuple[QCKQuery, List[KDP]]]:
    print("loading ranked list")
    ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(q_res_path)
    print("Pre loading docs")
    preload_docs(ranked_list, top_n)
    entries: List[Tuple[QCKQuery, List[KnowledgeDocumentPart]]] = []

    all_doc_parts = 0
    ticker = TimeEstimator(len(queries))
    for q in queries:
        q_res: List[GalagoDocRankEntry] = ranked_list[q.query_id]
        doc_part_list = enum_doc_parts_from_ranked_list(config, q_res, top_n)
        all_doc_parts += len(doc_part_list)
        entries.append((q, doc_part_list))
        ticker.tick()
    return entries


def enum_doc_parts_from_ranked_list(config, q_res, top_n):
    docs = iterate_docs(q_res, top_n)
    doc_part_list: List[KDP] = iterate_document_parts(docs, config['window_size'], config['step_size'])
    return doc_part_list


class QKWorker:
    def __init__(self, q_res_path, config, top_n):
        self.config = config
        self.top_n = top_n
        print("loading ranked list")
        self.ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(q_res_path)
        print("Ranked list loaded for {} queries".format(len(self.ranked_list)))
        print("Pre loading docs")
        preload_docs(self.ranked_list, top_n)

    def work(self, q: QCKQuery):
        all_doc_parts = 0
        q_res: List[GalagoDocRankEntry] = self.ranked_list[q.query_id]
        doc_part_list = enum_doc_parts_from_ranked_list(self.config, q_res, self.top_n)
        all_doc_parts += len(doc_part_list)
        return doc_part_list


def get_qk_candidate(config, q_res_path, qck_queries):
    top_n = config['top_n']
    worker = QKWorker(q_res_path, config, top_n)
    all_candidate = []
    ticker = TimeEstimator(len(qck_queries))
    for q in qck_queries:
        ticker.tick()
        try:
            doc_part_list = worker.work(q)
            e = q, doc_part_list
            all_candidate.append(e)
        except KeyError as e:
            print(e)
    return all_candidate