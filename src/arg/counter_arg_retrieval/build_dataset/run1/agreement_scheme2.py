import os
import sys
from collections import Counter
from typing import List, Any

from scipy.stats import pearsonr, kendalltau

from arg.counter_arg_retrieval.build_dataset.verify_common import summarize_agreement, print_hit_answers
from list_lib import lmap, foreach
from misc_lib import group_by, get_first, get_second, get_third, average
from mturk.parse_util import HITScheme, ColumnName, Checkbox, parse_file, HitResult, RadioButtonGroup
from stats.agreement import cohens_kappa

"""
          <div><crowd-checkbox name="Q1">Q1. The claim supports the topic. </crowd-checkbox></div>
          <div><crowd-checkbox name="Q2">Q2. The claim opposes the topic. </crowd-checkbox></div>

        <p>Q1,2: Does the claim support/oppose the topic? </p>

        <p>Q3: Is the document somehow <b>related</b> to the <u>topic</u> or the <u>claim</u>? </p>
        <p>&nbsp;&nbsp; ∘ If not, the answers for all the followings questions would be 'No' (probably). </p>
        <p>Q4: Does the document <b>mention</b> the given <u>topic</u>? </p>
        <p>&nbsp;&nbsp; ∘ If the document mentions the claim, the reader of the document would think about the topic when reading the document. </p>
        <p>Q5: Does the document <b>mention</b> the given <u>claim</u>? </p>
        <p>Q6,7: Does the document contain <b>arguments</b> that supports/opposes the <u>topic</u>?</p>
        <p>Q8,9: Does the document contain <b>arguments</b> that supports/opposes the <u>claim</u>?</p>
        <p>Q10,11: Does the document contain <b>information</b> that can be used to support/oppose the <u>topic</u>?</p>
        <p>Q12,13: Does the document contain <b>information</b> that can be used to support/oppose the <u>claim</u>?</p>
        <p>Q14: Overall, is the document useful to be recommended to the readers who want to know more about the <u>topic</u> or <u>claim</u>?</p>
        <ul>
            <li> 3. Very useful, the document is providing new critical and surprising information which would be important to be recommended to people who know the topic and the claim.</li>
            <li> 2. Useful, The document mentions some new information that is closely related to the topic or the claim. </li>
            <li> 1. The document has some related information, but that is mostly trivial repetition of the topic or the claim.</li>
            <li> 0. The document is not useful at all. </li>

        </ul>
"""
scheme2_question_d = {
    "Q1.on": "claim supports topic",
    "Q2.on": "claim opposes topic",
    "Q3.on": "document related (topic or claim)",
    "Q4.on": "document mentions topic",
    "Q5.on": "document mentions claim",
    "Q6.on": "document contains argument supporting topic",
    "Q7.on": "document contains argument opposing topic ",
    "Q8.on": "document contains argument supporting claim",
    "Q9.on": "document contains argument opposing claim",
    "Q10.on": "document contains information supporting topic",
    "Q11.on": "document contains information opposing topic ",
    "Q12.on": "document contains information supporting claim",
    "Q13.on": "document contains information opposing claim",
    "Q14.": "useful?"
}


def get_ca_run1_scheme2():
    inputs = [ColumnName("c_text"), ColumnName("p_text"), ColumnName("doc_id")]
    # answer_list = [
    #     "Q1.on",
    #     "Q14.0.Q14.0",
    #     "Q14.1.Q14.1",
    #     "Q14.2.Q14.2",
    #     "Q14.3.Q14.3",
    #     "Q2.on",
    #     "claim_arg_oppose.on",
    #     "claim_arg_support.on",
    #     "claim_info_oppose.on",
    #     "claim_info_support.on",
    #     "claim_mention.on",
    #     "related.on",
    #     "topic_arg_oppose.on",
    #     "topic_arg_support.on",
    #     "topic_info_oppose.on",
    #     "topic_info_support.on",
    #     "topic_mention.on",
    # ]
    answer_units = []
    for i in range(1, 14):
        answer_units.append(Checkbox("Q{}.on".format(i)))

    answer_units.append(RadioButtonGroup("Q14.", lmap(str, range(4)), True))
    hit_scheme = HITScheme(inputs, answer_units)
    return hit_scheme


def apply_transitive(hit: HitResult):

    claim_support_topic = hit.outputs["Q1.on"]
    claim_oppose_topic = hit.outputs["Q2.on"]

    argument_supporting_claim = hit.outputs["Q8.on"]
    argument_opposing_claim = hit.outputs["Q9.on"]
    if claim_support_topic and argument_supporting_claim:
        hit.outputs["Q6.on"] = 1 # arg support topic

    # if claim_support_topic and argument_opposing_claim:
    #     hit.outputs["Q7.on"] = 1 # arg oppose topic

    # if claim_oppose_topic and argument_supporting_claim:
    #     hit.outputs["Q7.on"] = 1 # arg support topic
    #
    # if claim_oppose_topic and argument_opposing_claim:
    #     hit.outputs["Q6.on"] = 1 # arg support topic




def show_agreement():
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    foreach(apply_transitive, hit_results)

    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        annot1: List[Any] = lmap(get_first, list_answers)
        annot2: List[Any] = lmap(get_second, list_answers)
        annot3: List[Any] = lmap(get_third, list_answers)
        k12 = cohens_kappa(annot1, annot2)
        k23 = cohens_kappa(annot2, annot3)
        k31 = cohens_kappa(annot3, annot1)
        avg_k = average([k12, k23, k31])
        print("{0}\t{1:.2f}\t{2}".format(answer_column, avg_k, scheme2_question_d[answer_column]))


def pearsonr_fixed(annot1, annot2):
    a, b = pearsonr(annot1, annot2)
    return "{0:.2f} {1:.2f}".format(a, b)


def kendalltau_fixed(annot1, annot2):
    a, b = kendalltau(annot1, annot2)
    return "{0:.2f} {1:.2f}".format(a, b)



def measure_correlation(hit_results: List[HitResult]):
    metric_fn = pearsonr_fixed
    answer_list_d = summarize_agreement(hit_results)
    for answer_column, list_answers in answer_list_d.items():
        annot1: List[Any] = lmap(get_first, list_answers)
        annot2: List[Any] = lmap(get_second, list_answers)
        annot3: List[Any] = lmap(get_third, list_answers)
        print(answer_column)
        print('1 vs 2', metric_fn(annot1, annot2))
        print('2 vs 3', metric_fn(annot2, annot3))
        print('3 vs 1', metric_fn(annot3, annot1))


def answers():
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    answer_list_d = summarize_agreement(hit_results)
    print(answer_list_d['key'])
    for answer_column, list_answers in answer_list_d.items():
        print(answer_column)
        print(list_answers)


def answers_per_input():
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(sys.argv[1], hit_scheme)
    print_hit_answers(hit_results)


def count_counter_argument():
    common_dir = "C:\\work\\Code\\Chair\\output\\ca_building\\run1\\mturk_output"
    file_path = os.path.join(common_dir,  "Batch_4490355_batch_results.csv")
    hit_scheme = get_ca_run1_scheme2()
    hit_results: List[HitResult] = parse_file(file_path, hit_scheme)
    input_columns = list(hit_results[0].inputs.keys())

    def get_hit_key(hit_result: HitResult):
        return tuple([hit_result.inputs[key] for key in input_columns])
    answer_columns = list(hit_results[0].outputs.keys())

    counter = Counter()
    for key, entries in group_by(hit_results, get_hit_key).items():
        answer_column = "Q13"

        def get_answers_for_column(answer_column):
            return list([e.outputs[answer_column] for e in entries])

        for c in answer_columns:
            answers = get_answers_for_column(c)
            if sum(answers) >= 2:
                counter[c] += 1

        if len(entries) >= 2:
            counter[c] += 1


def main():
    answers_per_input()


def do_measure_correlation():
    common_dir = "C:\\work\\Code\\Chair\\output\\ca_building\\run1\\mturk_output"
    file_path_2 = os.path.join(common_dir,  "Batch_4490355_batch_results.csv")
    hit_scheme = get_ca_run1_scheme2()
    for file_path in [file_path_2]:
        hit_results: List[HitResult] = parse_file(file_path, hit_scheme)
        measure_correlation(hit_results)


if __name__ == "__main__":
    show_agreement()
