from typing import Iterable

from arg.perspectives.ppnc.decl import ClaimPassages
from arg.qck.decl import QKInstance
from list_lib import flatten
from misc_lib import DataIDManager


def generate_instances(claim_passages_list: Iterable[ClaimPassages],
                       data_id_manager: DataIDManager) -> Iterable[QKInstance]:

    def convert(pair: ClaimPassages) -> Iterable[QKInstance]:
        claim, passages = pair
        cid = claim['cId']
        query_text = claim['text']
        for passage_idx, (passage, dummy_score) in enumerate(passages):
            info = {
                        'cid': cid,
                        'passage_idx': passage_idx
                    }
            yield QKInstance(query_text, passage, data_id_manager.assign(info))

    return flatten(map(convert, claim_passages_list))


