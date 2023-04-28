class InputShapeConfigTT:
    max_terms: int
    max_subword_per_word: int


class InputShapeConfigTT100_4(InputShapeConfigTT):
    max_terms = 100
    max_subword_per_word = 4