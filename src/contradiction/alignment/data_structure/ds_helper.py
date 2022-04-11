from typing import List

from contradiction.alignment.data_structure.eval_data_structure import RelatedEvalAnswer, ContributionSummary, \
    RelatedBinaryAnswer


def parse_related_eval_answer_from_json(raw_json) -> List[RelatedEvalAnswer]:
    def parse_entry(e) -> RelatedEvalAnswer:
        problem_id = e[0]
        assert type(problem_id) == str
        score_array_wrap = e[1]
        score_array = score_array_wrap[0]
        assert type(score_array) == list
        score_type = type(score_array[0][0])
        assert score_type == float or score_type == int
        return RelatedEvalAnswer(problem_id, ContributionSummary(score_array))
    return list(map(parse_entry, raw_json))


def parse_related_binary_answer_from_json(raw_json) -> List[RelatedBinaryAnswer]:
    def parse_entry(e) -> RelatedBinaryAnswer:
        problem_id = e[0]
        score_table = e[1]
        assert type(score_table[0][0]) == int
        return RelatedBinaryAnswer(problem_id, score_table)
    return list(map(parse_entry, raw_json))