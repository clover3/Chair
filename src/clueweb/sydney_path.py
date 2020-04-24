from typing import Set

clueweb12_B13_doc_id_to_url_path = "/mnt/nfs/collections/ClueWeb12/ClueWeb12-B13/ClueWeb12_B13_DocID_To_URL.txt"


def get_clueweb12_B13_doc_ids() -> Set[str]:
    f = open(clueweb12_B13_doc_id_to_url_path)
    s = set()
    for line in f:
        idx = line.find(",")
        doc_id = line[:idx]
        s.add(doc_id)

    return s
