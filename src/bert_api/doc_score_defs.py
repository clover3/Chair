from typing import NamedTuple, List, Tuple

from data_generator.tokenize_helper import SbwordIdx, WordIdx, TokenizedText


class DocumentScorerOutputSbword(NamedTuple):
    window_start_loc: List[SbwordIdx]  # In sbword
    scores: List[float]


class DocumentScorerOutput(NamedTuple):
    window_start_loc: List[WordIdx]  # In sbword
    scores: List[float]

    @classmethod
    def from_dsos(cls, document_scorer_output_subword: DocumentScorerOutputSbword, doc: TokenizedText):
        window_start_loc: List[WordIdx] = []
        for sbword_idx in document_scorer_output_subword.window_start_loc:
            word_idx = WordIdx(doc.sbword_mapping[sbword_idx])
            window_start_loc.append(word_idx)
        return DocumentScorerOutput(window_start_loc, document_scorer_output_subword.scores)

    def __len__(self):
        if len(self.window_start_loc) != len(self.scores):
            print(f"WARNING len(self.window_start_loc) != len(self.scores) :"
                  f" {len(self.window_start_loc)} != {len(self.scores) }")
        return len(self.window_start_loc)


class DocumentScorerInput(NamedTuple):
    window_start_loc: List[SbwordIdx]
    payload_list: List[Tuple[List[str], List[str]]]


class DocumentScorerInputEx(NamedTuple):
    window_start_loc: List[SbwordIdx]
    payload_list: List[Tuple[List[str], List[str]]]
    doc: TokenizedText


