from typing import List, Callable, Dict, Tuple
from typing import NamedTuple

from arg.perspectives.evaluate import perspective_getter
from arg.perspectives.load import PerspectiveCluster, load_perspectrum_golds
from list_lib import lmap_pairing, lmap


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


class CaTopicv2(NamedTuple):
    cid: str
    claim_text: str
    ca_cid: str
    target_p: List[Tuple[str, str]]
    other_ps: List[List[Tuple[str, str]]]

    @classmethod
    def from_ca_topic(cls,
                      c: CaTopic,
                      cid_to_pid_clusters: Callable[[str], List[List[str]]],
                      p_text_getter: Callable[[str], str]):
        pid_clusters: List[List[str]] = cid_to_pid_clusters(c.cid)
        target_p = None
        other_ps = []
        for pid_cluster in pid_clusters:
            joined_ps: List[Tuple[str, str]] = lmap_pairing(p_text_getter, pid_cluster)
            if c.pid in pid_cluster:
                target_p = joined_ps
            else:
                other_ps.append(joined_ps)
        assert target_p is not None
        return CaTopicv2(c.cid, c.claim_text, c.ca_cid, target_p, other_ps)

    def to_dict(self):
        d = {
            'cid': self.cid,
            'claim_text': self.claim_text,
            'ca_cid': self.ca_cid,
            'target_p': self.target_p,
            'other_ps': self.other_ps
        }
        return d


def get_ca2_converter():
    gold_d: Dict[int, List[PerspectiveCluster]] = load_perspectrum_golds()

    def cid_to_pid_clusters(cid: str) -> List[List[str]]:
        return list([lmap(str, c.perspective_ids) for c in gold_d[int(cid)]])

    def perspective_getter_str_ver(pid: str):
        return perspective_getter(int(pid))

    def convert(ca1_topic) -> CaTopicv2:
        ca2_topic = CaTopicv2.from_ca_topic(ca1_topic, cid_to_pid_clusters, perspective_getter_str_ver)
        return ca2_topic
    return convert