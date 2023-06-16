import json
import math
from collections import defaultdict, Counter
from typing import List, Iterable, Callable, Dict, Tuple, Set

from contradiction.medical_claims.token_tagging.batch_solver_common import ECCOutput
from contradiction.medical_claims.token_tagging.gpt_solver.index_span import IndexedSpan, find_all_as_index_span, strip_char_set
from utils.open_ai_api import OpenAIProxy
from typing import List, Tuple, Dict

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from iter_util import load_jsonl
from misc_lib import average


class GPTSolver(TokenScoringSolverIF):
    def __init__(
            self,
            open_ai_proxy: OpenAIProxy,
            prompt_template,
            claim2_pattern,
            log_path,
            parse_gpt_response_fn,
            parse_answer_texts: Callable[[str], Tuple[str, str]]
    ):
        self.proxy: OpenAIProxy = open_ai_proxy
        self.prompt_template = prompt_template
        self.log_file = open(log_path, "a")
        self.claim2_pattern = claim2_pattern
        self.parse_gpt_response_fn = parse_gpt_response_fn
        self.parse_answer_texts = parse_answer_texts

    def solve(self, tokens1, tokens2) -> ECCOutput:
        claim1 = " ".join(tokens1)
        claim2 = " ".join(tokens2)
        prompt: str = self.prompt_template.format(claim1, claim2)
        j = self.proxy.request(prompt)
        self.log_file.write(json.dumps(j) + "\n")
        completion_text = self.parse_gpt_response_fn(j)
        claim1_answer, claim2_answer = self.parse_answer_texts(completion_text)
        return get_score_from_answer_spans(tokens1, tokens2, claim1_answer, claim2_answer)


class GPTRequester(TokenScoringSolverIF):
    def __init__(self,
                 open_ai_proxy: OpenAIProxy,
                 prompt_template,
                 log_path,
                 ):
        self.proxy: OpenAIProxy = open_ai_proxy
        self.prompt_template = prompt_template
        self.log_file = open(log_path, "a")

    def solve(self, tokens1, tokens2) -> ECCOutput:
        claim1 = " ".join(tokens1)
        claim2 = " ".join(tokens2)
        prompt: str = self.prompt_template.format(claim1, claim2)
        j = self.proxy.request(prompt)
        j_save = {
            'claim1': claim1,
            'claim2': claim2,
            'reponse': j
        }
        self.log_file.write(json.dumps(j_save) + "\n")
        scores1 = [0 for _ in tokens1]
        scores2 = [0 for _ in tokens2]
        return scores1, scores2


def load_json_log(log_path) -> Dict[Tuple[str, str], Dict]:
    j_d = {}
    for j in load_jsonl(log_path):
        key = j['claim1'], j['claim2']
        j_d[key] = j['reponse']
    return j_d


class GPTSolverFileRead(TokenScoringSolverIF):
    def __init__(
            self,
            j_d: Dict[Tuple[str, str], Dict],
            parse_gpt_response_fn: Callable[[Dict], str],
            parse_answer_texts: Callable[[str], Tuple[str, str]]
    ):
        self.parse_gpt_response_fn = parse_gpt_response_fn
        self.parse_answer_texts = parse_answer_texts
        self.j_d = j_d

    def solve(self, tokens1, tokens2) -> ECCOutput:
        claim1 = " ".join(tokens1)
        claim2 = " ".join(tokens2)
        j_response = self.j_d[claim1, claim2]
        completion_text = self.parse_gpt_response_fn(j_response)
        claim1_answer, claim2_answer = self.parse_answer_texts(completion_text)
        return get_score_from_answer_spans(tokens1, tokens2, claim1_answer, claim2_answer)


def get_parse_answer_texts_for_instruct(prompt, claim2_pattern):
    def parse(completion_text):
        return parse_answer_texts_from_completion_text(prompt, claim2_pattern, completion_text)
    return parse


def get_score_from_answer_spans(
        tokens1: List[str], tokens2: List[str],
        claim1_answer: str, claim2_answer: str) -> Tuple[List[float], List[float]]:
    claim1 = " ".join(tokens1)
    claim2 = " ".join(tokens2)
    score_d1 = assign_scores_from_text(claim1, claim1_answer)
    score_d2 = assign_scores_from_text(claim2, claim2_answer)

    def d_to_arr(d: Dict, l: int) -> List[float]:
        scores: List[float] = [0 for _ in range(l)]
        for i, f in d.items():
            scores[i] = f
        return scores

    scores1: List[float] = d_to_arr(score_d1, len(tokens1))
    scores2: List[float] = d_to_arr(score_d2, len(tokens2))
    return scores1, scores2


def parse_answer_texts_from_completion_text(prompt, claim2_pattern, completion_text) -> Tuple[str, str]:
    full_text = prompt + completion_text
    claim2_line_st: int = full_text.lower().find(claim2_pattern.lower())
    if claim2_line_st < 0:
        print("Fail to parse: ", completion_text)
        raise IndexError()
    claim2_answer_st: int = claim2_line_st + len(claim2_pattern)
    claim2_answer_ed = len(full_text)
    claim2_answer = IndexedSpan(full_text, claim2_answer_st, claim2_answer_ed)
    # Identify location for each claim's answer
    claim1_answer_st = len(prompt)
    claim1_answer_ed = claim2_line_st
    claim1_answer = IndexedSpan(full_text, claim1_answer_st, claim1_answer_ed)
    if not claim1_answer.to_text().strip():
        raise ValueError()
    if not claim2_answer.to_text().strip():
        raise ValueError()
    print("claim1_answer", claim1_answer.to_text())
    print("claim2_answer", claim2_answer.to_text())
    return claim1_answer.to_text(), claim2_answer.to_text()


def get_offset_to_prob(full_text, text_offset, token_logprobs, tokens):
    offset_to_prob = {}
    for token, logprob, offset in zip(tokens, token_logprobs, text_offset):
        if token == "<|endoftext|>":
            break
        token_ = full_text[offset:offset + len(token)]
        assert token == token_
        if full_text[offset].isspace():
            offset = offset + 1
        offset_to_prob[offset] = math.exp(logprob)
    return offset_to_prob


def guess_delimiter(text):
    options = [",", ";", "/"]
    counter = Counter()
    for ch in options:
        n_try = len(text.split(ch))
        counter[ch] = n_try

    ch_max, n = counter.most_common(1)[0]
    if n >= 2:
        return ch_max
    return ","


def align_scores(claim: str, claim_answer: IndexedSpan, offset_to_prob: Dict):
    score_d = {}
    delimiter = guess_delimiter(claim_answer.to_text())
    print('claim', claim)
    print("Use {} as delimiter".format(delimiter))
    for raw_chunk in claim_answer.split(delimiter):
        chunk: IndexedSpan = raw_chunk.strip().strip_quotation()
        tokens: List[IndexedSpan] = chunk.split()
        if not tokens:
            raise IndexError("There is no token in chunk")
        print("chunk", chunk.to_text())
        token_level_score_assign(claim, offset_to_prob, score_d, tokens)
    return score_d


def assign_scores(claim: str, claim_answer: IndexedSpan):
    def token_norm(t) -> str:
        strip_ch_set = " .,;'!?\"\'{}()"
        st, ed = strip_char_set(t.lower(), 0, len(t), strip_ch_set)
        return t.lower()[st:ed]

    c_tokens = [token_norm(t) for t in claim.split()]
    delimiter = guess_delimiter(claim_answer.to_text())
    print('claim', claim)
    print("Use {} as delimiter".format(delimiter))
    mismatch_words = set()
    for raw_chunk in claim_answer.split(delimiter):
        chunk_text = token_norm(raw_chunk.to_text())
        for t in chunk_text.split():
            mismatch_words.add(token_norm(t))

    score_d = {}
    for i, t in enumerate(c_tokens):
        if t in mismatch_words:
            score_d[i] = 1
        else:
            score_d[i] = 0

    n_common = sum(score_d.values())
    n_gpt = len(mismatch_words)
    if n_common < n_gpt:
        print("GPT has output {} tokens but {} were matched".format(n_gpt, n_common))
        print("claim tokens:", c_tokens)
        print("GPT tokens:", mismatch_words)

    return score_d


def assign_scores_from_text(claim: str, claim_answer: str):
    def token_norm(t) -> str:
        strip_ch_set = " .,;'!?\"\'{}()"
        st, ed = strip_char_set(t.lower(), 0, len(t), strip_ch_set)
        return t.lower()[st:ed]

    c_tokens = [token_norm(t) for t in claim.split()]
    delimiter = guess_delimiter(claim_answer)
    print('claim', claim)
    print("Use {} as delimiter".format(delimiter))
    mismatch_words = set()
    for raw_chunk in claim_answer.split(delimiter):
        chunk_text = token_norm(raw_chunk)
        for t in chunk_text.split():
            mismatch_words.add(token_norm(t))

    score_d = {}
    for i, t in enumerate(c_tokens):
        if t in mismatch_words:
            score_d[i] = 1
        else:
            score_d[i] = 0

    n_common = sum(score_d.values())
    n_gpt = len(mismatch_words)
    if n_common < n_gpt:
        print("GPT has output {} tokens but {} were matched".format(n_gpt, n_common))
        print("claim tokens:", c_tokens)
        print("GPT tokens:", mismatch_words)

    return score_d


def token_level_score_assign(claim, offset_to_prob, score_d,
                             tokens: List[IndexedSpan]):
    score_d_local: Dict[int, List[float]] = defaultdict(list)
    n_not_found = 0
    for token in tokens:
        span_list: List[IndexedSpan] = find_all_as_index_span(claim, token.to_text())
        if not span_list:
            n_not_found += 1
        for span_in_claim in span_list:
            indices: List[int] = span_in_claim.get_sp_token_indices()
            print(indices, [str(t) for t in tokens])
            prob = offset_to_prob[token.st]
            for index in indices:
                score_d_local[index].append(prob)

    if n_not_found > len(tokens) * 0.7:
        raise IndexError("{} of {} tokens are not matched".format(n_not_found, len(tokens)))

    for index, scores in score_d_local.items():
        score_d[index] = average(scores)


def span_level_score_assign(chunk, claim, offset_to_prob, score_d, tokens):
    span_list: List[IndexedSpan] = find_all_as_index_span(claim, chunk.to_text())
    if not span_list:
        raise IndexError("Span are not found")
    for span_in_claim in span_list:
        indices: List[int] = span_in_claim.get_sp_token_indices()
        print(indices, [str(t) for t in tokens])
        assert len(indices) == len(tokens)
        for index, token in zip(indices, tokens):
            prob = offset_to_prob[token.st - 1]
            assert index not in score_d
            score_d[index] = prob


