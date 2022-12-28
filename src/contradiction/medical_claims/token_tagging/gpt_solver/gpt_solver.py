import json
import math
from collections import defaultdict, Counter

from contradiction.medical_claims.token_tagging.batch_solver_common import BatchTokenScoringSolverIF, ECCOutput, \
    ECCInput
from contradiction.medical_claims.token_tagging.gpt_solver.index_span import strip_index, IndexedSpan, split_indexed, \
    find_all, find_all_as_index_span, strip_char_set
from contradiction.medical_claims.token_tagging.gpt_solver.open_ai_api import OpenAIProxy
from typing import List, Tuple, Dict

from contradiction.medical_claims.token_tagging.online_solver_common import TokenScoringSolverIF
from cpath import output_path
from iter_util import load_jsonl
from misc_lib import path_join, average


class GPTSolver(TokenScoringSolverIF):
    def __init__(self, open_ai_proxy: OpenAIProxy,
                 prompt_template,
                 claim2_pattern,
                 log_path,
                 ):
        self.proxy: OpenAIProxy = open_ai_proxy
        self.prompt_template = prompt_template
        self.log_file = open(log_path, "a")
        self.claim2_pattern = claim2_pattern

    def solve(self, tokens1, tokens2) -> ECCOutput:
        claim1 = " ".join(tokens1)
        claim2 = " ".join(tokens2)
        prompt: str = self.prompt_template.format(claim1, claim2)
        j = self.proxy.request(prompt)
        self.log_file.write(json.dumps(j) + "\n")
        score_pair: ECCOutput = get_score_from_j(prompt, tokens1, tokens2, j, self.claim2_pattern)
        return score_pair


class GPTRequester(TokenScoringSolverIF):
    def __init__(self, open_ai_proxy: OpenAIProxy,
                 prompt_template,
                 claim2_pattern,
                 log_path,
                 ):
        self.proxy: OpenAIProxy = open_ai_proxy
        self.prompt_template = prompt_template
        self.log_file = open(log_path, "w")
        self.claim2_pattern = claim2_pattern

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


class GPTSolverFileRead(TokenScoringSolverIF):
    def __init__(self,
                 prompt_template,
                 claim2_pattern,
                 log_path,
                 ):
        self.prompt_template = prompt_template
        self.claim2_pattern = claim2_pattern

        j_d = {}
        for j in load_jsonl(log_path):
            key = j['claim1'], j['claim2']
            j_d[key] = j['reponse']
        self.j_d = j_d


    def solve(self, tokens1, tokens2) -> ECCOutput:
        claim1 = " ".join(tokens1)
        claim2 = " ".join(tokens2)
        prompt: str = self.prompt_template.format(claim1, claim2)
        j_response = self.j_d[claim1, claim2]
        return get_score_from_j(prompt, tokens1, tokens2,
                     j_response, self.claim2_pattern)


def get_score_from_j(prompt, tokens1, tokens2,
                     j_response, claim2_pattern):
    claim1 = " ".join(tokens1)
    claim2 = " ".join(tokens2)
    choice = j_response['choices'][0]
    completion_text = choice['text']
    logprobs = choice['logprobs']
    tokens = logprobs['tokens']
    token_logprobs = logprobs['token_logprobs']
    text_offset = logprobs['text_offset']
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

    # claim1_answer: str = full_text[len(prompt):claim2_line_st]
    # claim2_answer: str = full_text[claim2_answer_st:]
    print("claim1_answer", claim1_answer.to_text())
    print("claim2_answer", claim2_answer.to_text())
    assert len(tokens) == len(token_logprobs)
    assert len(tokens) == len(text_offset)

    # Record probability
    # offset_to_prob: Dict[int, float] = get_offset_to_prob(full_text, text_offset, token_logprobs, tokens)

    score_d1 = assign_scores(claim1, claim1_answer)
    score_d2 = assign_scores(claim2, claim2_answer)

    def d_to_arr(d: Dict, l: int) -> List[float]:
        scores: List[float] = [0 for _ in range(l)]
        for i, f in d.items():
            scores[i] = f
        return scores

    scores1: List[float] = d_to_arr(score_d1, len(tokens1))
    scores2: List[float] = d_to_arr(score_d2, len(tokens2))
    return scores1, scores2


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


def get_mismatch_prediction_prompt_template():
    instruction = "In each of the examples, " \
                  "two claims extracted from research paper abstracts will be shown. " \
                  "The given two claims seem to be contradictory as they are implying" \
                  " opposite results about the same question. " \
                  "Precisely though, the two claims may have been obtained" \
                  " for different population or intervention details " \
                  "that make it possible that both claims to be true." \
                  " We want to annotate the tokens (words) that" \
                  " express different conditions."

    problem = "Claim 1: {}\nClaim 2: {}"
    later_template = "Condition tokens in Claim 1:"
    return instruction + "\n\n" + problem + "\n\n" + later_template


def get_conflict_prediction_prompt_template():
    instruction = "In each of the examples, " \
                  "two claims extracted from research paper abstracts will be shown. " \
                  "The given two claims seem to be contradictory as they are implying" \
                  " opposite results about the same question. " \
                  "Precisely though, the two claims may have been obtained" \
                  " for different population or intervention details " \
                  "that make it possible that both claims to be true." \
                  " We want to annotate the tokens (words) that" \
                  " express opposite results."

    problem = "Claim 1: {}\nClaim 2: {}"
    later_template = "Opposite results tokens in Claim 1:"
    return instruction + "\n\n" + problem + "\n\n" + later_template


def get_gpt_solver_mismatch() -> GPTSolver:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_mismatch.json")
    return GPTSolver(OpenAIProxy("text-davinci-003"),
                     get_mismatch_prediction_prompt_template(),
                     "Condition tokens in Claim 2:",
                     log_path)


def get_gpt_requester_mismatch() -> GPTRequester:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_mismatch.json")
    return GPTRequester(OpenAIProxy("text-davinci-003"),
                     get_mismatch_prediction_prompt_template(),
                     "Condition tokens in Claim 2:",
                     log_path)


def get_gpt_requester_conflict() -> GPTRequester:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_conflict.json")
    return GPTRequester(OpenAIProxy("text-davinci-003"),
                     get_conflict_prediction_prompt_template(),
                     "Opposite results tokens in Claim 2:",
                     log_path)



def get_gpt_file_solver_mismatch() -> GPTSolverFileRead:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_mismatch.mod.json")
    return GPTSolverFileRead(get_mismatch_prediction_prompt_template(),
                     "Condition tokens in Claim 2:",
                     log_path)


def get_gpt_file_solver_conflict() -> GPTSolverFileRead:
    log_path = path_join(output_path, "alamri_annotation1", "gpt", "davinci_req_conflict.mod.json")
    return GPTSolverFileRead(get_conflict_prediction_prompt_template(),
                     "Opposite results tokens in Claim 2",
                     log_path)
