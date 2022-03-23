import csv
from typing import List
from typing import NamedTuple


class CAQuery(NamedTuple):
    qid: str
    claim: str
    perspective: str
    ca_query: str
    def summary(self):
        return "QID: {}\n" "Claim:{}\n" "Perspective: {}\n" "CAQuery: {}".format(
            self.qid, self.claim, self.perspective, self.ca_query)


class CATask(NamedTuple):
    qid: str
    claim: str
    c_stance: str
    perspective: str
    p_stance: str
    pc_stance: str
    entity: str

def load_ca_query_from_tsv(file_path) -> List[CAQuery]:
    f = open(file_path)
    outputs = []
    for row in csv.reader(f, delimiter="\t"):
        qid = row[0]
        claim = row[1]
        perspective = row[2]
        ca_query = row[3]
        e = CAQuery(qid, claim, perspective, ca_query)
        outputs.append(e)
    return outputs


