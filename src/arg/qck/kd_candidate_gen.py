from typing import List, Iterable, Dict, Tuple

from arg.perspectives.clueweb_db import load_doc
from arg.qck.decl import QCKQuery, KnowledgeDocument, KnowledgeDocumentPart, KDP
from datastore.interface import preload_man
from datastore.table_names import TokenizedCluewebDoc
from galagos.parse import load_galago_ranked_list
from galagos.types import GalagoDocRankEntry


# KD = Knowledge Document

def preload_docs(ranked_list: Dict[str, List[GalagoDocRankEntry]], top_n):
    all_doc_ids = set()
    for entries in ranked_list.values():
        for entry in entries[:top_n]:
            all_doc_ids.add(entry.doc_id)

    print(f"total of {len(all_doc_ids)} docs")
    print("Accessing DB")
    #  Get the doc from DB
    preload_man.preload(TokenizedCluewebDoc, all_doc_ids)


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
            print("doc len", len(tokens))
            kd = KnowledgeDocument(q_res[i].doc_id, tokens)
            docs.append(kd)
        except KeyError:
            pass

    if len(docs) < top_n:
        print("Retrieved {} of {} docs".format(len(docs), top_n))

    duplicate_doc_ids = get_duplicate(docs)
    unique_docs = [d for d in docs if d.doc_id not in duplicate_doc_ids]
    return unique_docs


def iterate_document_parts(docs: Iterable[KnowledgeDocument]) -> List[KnowledgeDocumentPart]:
    # knowledge document parts list
    kdp_list: List[KnowledgeDocumentPart] = []
    for doc in docs:
        idx = 0
        passage_idx = 0
        window_size = 300
        while idx < len(doc.tokens):
            tokens = doc.tokens[idx:idx + window_size]
            kdp = KnowledgeDocumentPart(doc.doc_id, passage_idx, idx, tokens)
            kdp_list.append(kdp)
            idx += window_size
            print(idx)
            passage_idx += 1
    return kdp_list


def qk_candidate_gen(q_res_path: str, queries: List[QCKQuery], top_n) -> List[Tuple[QCKQuery, List[KDP]]]:
    ranked_list: Dict[str, List[GalagoDocRankEntry]] = load_galago_ranked_list(q_res_path)
    preload_docs(ranked_list, top_n)
    entries: List[Tuple[QCKQuery, List[KnowledgeDocumentPart]]] = []

    all_doc_parts = 0
    for q in queries:
        q_res: List[GalagoDocRankEntry] = ranked_list[q.query_id]
        docs = iterate_docs(q_res, top_n)
        doc_part_list: List[KDP] = iterate_document_parts(docs)
        all_doc_parts += len(doc_part_list)
        entries.append((q, doc_part_list))

    return entries

