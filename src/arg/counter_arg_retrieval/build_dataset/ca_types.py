from typing import NamedTuple


class CaTopic(NamedTuple):
    cid: str
    claim_text: str
    ca_cid: str
    p_text: str
    pid: str
    stance: str

    @classmethod
    def from_j_entry(self, j):
        return CaTopic(str(j['cid']),
                       j['claim_text'],
                       str(j['ca_cid']),
                       j['perspective']['p_text'],
                       str(j['perspective']['pid']),
                       j['perspective']['stance'])
