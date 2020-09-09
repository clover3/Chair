from typing import List, Dict, Tuple
from typing import NewType

CIDtoPassages = NewType('CIDtoPassages', Dict[int, List[Tuple[List[str], float]]])
ClaimPassages = NewType('ClaimPassages', Tuple[Dict, List[Tuple[List[str], float]]])
