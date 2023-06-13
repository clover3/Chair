from dataclasses import dataclass, field
from typing import List, Union
from typing import List, Iterable, Callable, Dict, Tuple, Set


@dataclass
class Token:
    id: str
    text: str


@dataclass
class Facet:
    id: str
    govNum: str
    modNum: str
    reltn: str
    govText: str
    modText: str
    childProp1Facet1: str = None
    childProp2Facet1: str = None

    def is_valid(self):
        exclude_terms = ["Topic", "Agent", "Root",
                        "Cause", "Quantifier", "neg"]
        for t in exclude_terms:
            if t in self.reltn:
                return False
        return True


@dataclass
class ReferenceAnswer:
    id: str
    text: str
    tokens: List[Token] = field(default_factory=list)
    facets: List[Facet] = field(default_factory=list)

    def __init__(self, id, text, tokens, facets):
        self.id = id
        self.text = text
        self.tokens = tokens
        self.facets = facets
        self.token_indices = {}
        for idx, token in enumerate(self.tokens):
            self.token_indices[token.id] = idx

    def facet_location(self, facet: Facet) -> Tuple[int, int]:
        loc1 = self.get_token_index(facet.govNum)
        loc2 = self.get_token_index(facet.modNum)
        return loc1, loc2

    def get_token_index(self, token_id) -> Union[int, None]:
        try:
            return self.token_indices[token_id]
        except KeyError:
            if token_id.endswith("_ra0"):  # Root
                return None
            else:
                raise



Expressed = "Expressed"
Unaddressed = "Unaddressed"


@dataclass
class FacetEntailment:
    facet_id: str
    label: str

    def get_bool_label(self):
        if self.label == Expressed:
            return True
        elif self.label == Unaddressed:
            return False
        else:
            raise ValueError()


@dataclass
class StudentAnswer:
    id: str
    answer_text: str
    accuracy: str
    facet_entailments: List[FacetEntailment] = field(default_factory=list)


# Extend Question class to include student answers
@dataclass
class Question:
    id: str
    module: str
    question_text: str
    reference_answer: ReferenceAnswer
    student_answers: List[StudentAnswer] = field(default_factory=list)  # Add this line


@dataclass
class PTEPredictionPerFacet:
    facet_id: str
    score: float
    pred: bool


@dataclass
class PTEPredictionPerStudentAnswer:
    id: str
    facet_pred: List[PTEPredictionPerFacet] = field(default_factory=list)


@dataclass
class PTEPredictionPerQuestion:
    id: str
    per_student_answer_list: List[PTEPredictionPerStudentAnswer] = field(default_factory=list)


