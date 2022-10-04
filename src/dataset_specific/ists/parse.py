import xml.etree.ElementTree as ET
from typing import List, Tuple
from typing import NamedTuple


class AlignmentLabelUnit(NamedTuple):
    chunk_token_id1: List[int]
    chunk_token_id2: List[int]
    chunk_text1: str
    chunk_text2: str
    align_types: List[str]
    align_score: str

    def serialize(self) -> str:
        s0_0 = " ".join(map(str, self.chunk_token_id1))
        s0_1 = " ".join(map(str, self.chunk_token_id2))

        s0 = f"{s0_0} <==> {s0_1}"
        s1 = self.align_types
        s2 = str(self.align_score)
        s3 = f"{self.chunk_text1} <==> {self.chunk_text2}"
        return " // ".join([s0, s1, s2, s3])

AlignmentPredictionList = List[Tuple[str, List[AlignmentLabelUnit]]]


class ISTSProblem(NamedTuple):
    problem_id: str
    text1: str
    text2: str


def parse_alignment_line(line) -> AlignmentLabelUnit:
    segments = line.split("//")
    chunk_token_id1, chunk_token_id2 = segments[0].split("==")
    align_type = segments[1].split("_")
    align_score = segments[2]
    chunk_token_id1 = [int(idx) for idx in chunk_token_id1.split()]
    chunk_token_id2 = [int(idx) for idx in chunk_token_id2.split()]
    chunk_text1, chunk_text2 = segments[-1].split("==")
    chunk_text1, chunk_text2 = chunk_text1.strip(), chunk_text2.strip()
    alu = AlignmentLabelUnit(chunk_token_id1, chunk_token_id2, chunk_text1, chunk_text2, align_type, align_score)
    return alu


def parse_label_file(file_path) -> List[Tuple[str, List[AlignmentLabelUnit]]]:
    print(file_path)
    f = open(file_path, "r")
    alignments = f.read()
    alignments = alignments.replace("<==>", "==").replace("&", "&amp;")
    alignments = alignments.encode("ascii", errors="ignore")
    tree = ET.fromstring(alignments)
    sentences = tree.findall("sentence")

    output: List[Tuple[str, List[AlignmentLabelUnit]]] = []
    for idx, sentence in enumerate(sentences):
        problem_id = sentence.get("id")
        sentence_text = sentence.text.split("\n")[1:-1]
        sentence_alignment: str = sentence.findall("alignment")[0].text.lower()
        lines = sentence_alignment.split("\n")
        lines = [line for line in lines if line.strip()]
        alu_list = list(map(parse_alignment_line, lines))
        output.append((problem_id, alu_list))

    return output