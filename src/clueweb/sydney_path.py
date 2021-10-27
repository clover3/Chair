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


index_list = ["/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk1_00.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk1_01.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk1_02.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk1_03.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk1_04.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk2_05.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk2_06.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk2_07.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk2_08.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk2_09.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk3.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk4_15.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk4_16.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk4_17.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk4_18.idx",
              "/mnt/lustre/godzilla/harding/ClueWeb12/idx/ClueWeb12-Disk4_19.idx"]