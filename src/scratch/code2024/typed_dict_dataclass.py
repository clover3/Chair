import dataclasses

import numpy as np


@dataclasses.dataclass
class QDWithAttndev:
    query: str
    doc: str
    score: float




l = [
("Query", "Doc", 10.),
("Query", "Doc", 10.)
]

for row in l:
    a = QDWithAttndev(*row)
    print(a)

