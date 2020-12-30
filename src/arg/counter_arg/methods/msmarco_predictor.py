from typing import List, Callable

from arg.counter_arg.header import Passage
from bert_api.client_lib import get_ingham_client
from misc_lib import NamedNumber


def get_scorer() -> Callable[[Passage, List[Passage]], List[NamedNumber]]:
    client = get_ingham_client()

    def scorer(query_p: Passage, candidate: List[Passage]) -> List[NamedNumber]:
        payload = []
        text1 = query_p.text.splitlines()[0]
        print(text1)
        for c in candidate:
            payload.append((text1, c.text))

        r = client.request_multiple(payload)
        r = [NamedNumber(v, "") for v in r]
        return r

    return scorer
