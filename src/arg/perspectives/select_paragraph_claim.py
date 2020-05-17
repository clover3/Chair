from typing import List, Iterable, Callable, Dict, Any

from arg.perspectives.ranked_list_interface import StaticRankedListInterface
from arg.pf_common.base import Paragraph
from galagos.types import GalagoDocRankEntry


def select_paragraph_dp_list(ci: StaticRankedListInterface,
                             clue12_13_df,
                             paragraph_iterator: Callable[[GalagoDocRankEntry], Iterable[Paragraph]],
                             claim_List: List[Dict]) -> List[Any]:


    cdf = 50 * 1000 * 1000
