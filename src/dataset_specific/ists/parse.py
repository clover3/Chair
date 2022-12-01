import xml.etree.ElementTree as ET
from typing import List, Tuple
from typing import NamedTuple

from alignment.base_ds import TextPairProblem
from list_lib import list_equal
from misc_lib import read_non_empty_lines_stripped

ALIGN_EQUI = "EQUI"
ALIGN_OPPO = "OPPO"
# SPE1: both chunks have similar meanings, but chunk in sentence 1 is more specific.
ALIGN_SPE1 = "SPE1"  # Left is more specific
ALIGN_SPE2 = "SPE2"  # Right is more specific

ALIGN_SIMI = "SIMI"
ALIGN_REL = "REL"
ALIGN_NOALI = "NOALI"
type_list = [ALIGN_EQUI, ALIGN_OPPO, ALIGN_SPE1, ALIGN_SPE2, ALIGN_SIMI, ALIGN_REL, ALIGN_NOALI]


def types_to_str(types):
    if len(types) == 2:
        return "_".join(types)
    elif len(types) == 1:
        return types[0]
    else:
        assert False


class AlignmentLabelUnit(NamedTuple):
    chunk_token_id1: List[int]
    chunk_token_id2: List[int]
    chunk_text1: str
    chunk_text2: str
    align_types: List[str]
    align_score: int

    def serialize(self) -> str:
        s0_0 = " ".join(map(str, self.chunk_token_id1))
        s0_1 = " ".join(map(str, self.chunk_token_id2))

        s0 = f"{s0_0} <==> {s0_1}"
        s1 = types_to_str(self.align_types)
        s2 = str(self.align_score)
        s3 = f"{self.chunk_text1} <==> {self.chunk_text2}"
        return " // ".join([s0, s1, s2, s3])

    def sort_key(self):
        def zero_to_inf(v):
            if v== 0:
                return 100000
            else:
                return v

        return zero_to_inf(self.chunk_token_id1[0]), zero_to_inf(self.chunk_token_id2[0])

AlignmentPrediction = Tuple[str, List[AlignmentLabelUnit]]
AlignmentPredictionList = List[AlignmentPrediction]


class iSTSProblem(NamedTuple):
    problem_id: str
    text1: str
    text2: str

    def to_text_pair_problem(self):
        return TextPairProblem(self.problem_id, self.text1, self.text2)


class iSTSProblemWChunk(NamedTuple):
    problem_id: str
    text1: str
    text2: str
    chunks1: List[str]
    chunks2: List[str]
    chunk_tokens_ids1: List[List[int]]
    chunk_tokens_ids2: List[List[int]]

    def to_text_pair_problem(self):
        return TextPairProblem(self.problem_id, self.text1, self.text2)


def parse_align_type(s):
    s = s.strip()
    if "_" in s:
        s1, s2 = s.split("_")
        assert s2 in ["POL", "FACT"]
        return [s1, s2]
    else:
        return [s]


def parse_score(s) -> int:
    try:
        return int(s)
    except ValueError as e:
        if s.strip() == "NIL":
            return 0
        else:
            raise


def parse_alignment_line(line) -> AlignmentLabelUnit:
    segments = line.split("//")
    chunk_token_id1, chunk_token_id2 = segments[0].split("==")

    align_types = parse_align_type(segments[1])
    align_score: int = parse_score(segments[2])
    chunk_token_id1 = [int(idx) for idx in chunk_token_id1.split()]
    chunk_token_id2 = [int(idx) for idx in chunk_token_id2.split()]
    chunk_text1, chunk_text2 = segments[-1].split("==")
    chunk_text1, chunk_text2 = chunk_text1.strip(), chunk_text2.strip()
    alu = AlignmentLabelUnit(chunk_token_id1, chunk_token_id2, chunk_text1, chunk_text2, align_types, align_score)
    return alu


def parse_label_file(file_path) -> List[Tuple[str, List[AlignmentLabelUnit]]]:
    f = open(file_path, "r")
    alignments = f.read()
    if "<root>" not in alignments:
        alignments = "<root>" + alignments + "</root>"
    alignments = alignments.replace("<==>", "==").replace("&", "&amp;")
    alignments = alignments.encode("ascii", errors="ignore")
    tree = ET.fromstring(alignments)
    sentences = tree.findall("sentence")

    output: List[Tuple[str, List[AlignmentLabelUnit]]] = []
    for idx, sentence in enumerate(sentences):
        problem_id = sentence.get("id")
        sentence_text = sentence.text.split("\n")[1:-1]
        sentence_alignment: str = sentence.findall("alignment")[0].text
        lines = sentence_alignment.split("\n")
        lines = [line for line in lines if line.strip()]
        alu_list = list(map(parse_alignment_line, lines))
        output.append((problem_id, alu_list))

    return output


def load_ists_problem_w_path(sent1_path, sent2_path):
    sent1_list: List[str] = read_non_empty_lines_stripped(sent1_path)
    sent2_list: List[str] = read_non_empty_lines_stripped(sent2_path)
    assert len(sent1_list) == len(sent2_list)

    p_list = []
    for i in range(len(sent1_list)):
        problem_id = str(i+1)
        p = iSTSProblem(problem_id, sent1_list[i], sent2_list[i])
        p_list.append(p)
    return p_list

def parse_chunks(file_path):
    lines = open(file_path, "r").readlines()
    def parse_line(line) -> List[str]:
        line = line.strip()
        chunks = line.split("] [")
        assert chunks[0][0] == "["
        chunks[0] = chunks[0][1:]
        if not chunks[-1][-1] == "]":
            print(chunks)
            raise Exception()
        chunks[-1] = chunks[-1][:-1]
        for t in chunks:
            assert "[" not in t
            assert "]" not in t

        chunks = [t.strip() for t in chunks]
        return chunks
    ret = list(map(parse_line, lines))
    return ret


def equal_tokenized(t1, t2):
    f = list_equal(t1.split(), t2.split())
    if not f:
        print(t1)
        print(t2)
        raise Exception()


def join_problem_w_chunks(problems, sent1_list, sent2_list):
    p_list = []
    n_problem = len(sent1_list)
    for i in range(n_problem):
        p: iSTSProblem = problems[i]
        chunks1 = sent1_list[i]
        chunks2 = sent2_list[i]
        equal_tokenized(" ".join(chunks1),  p.text1)
        equal_tokenized(" ".join(chunks2),  p.text2)

        def get_chunk_ids(chunks) -> List[List[int]]:
            idx = 1
            ids = []
            for chunk in chunks:
                n_tokens = len(chunk.split())
                ids.append([j + idx for j in range(n_tokens)])
                idx = idx + n_tokens
            return ids

        p_new = iSTSProblemWChunk(p.problem_id, p.text1, p.text2,
                                  chunks1, chunks2,
                                  get_chunk_ids(chunks1), get_chunk_ids(chunks2))
        p_list.append(p_new)
    return p_list