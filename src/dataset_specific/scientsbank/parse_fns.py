from dataclasses import dataclass
from typing import List
from xml.etree import ElementTree as ET

from dataset_specific.scientsbank.path_helper import get_pte_dataset
from dataset_specific.scientsbank.pte_data_types import Question, ReferenceAnswer, Token, Facet, StudentAnswer, \
    FacetEntailment, Expressed, Unaddressed
from misc_lib import get_dir_files
import os


@dataclass
class SplitSpec:
    split_name: str
    use_subset: bool
    subset_portion: float = 1

    def get_save_name(self):
        if self.use_subset:
            return "{}_{}".format(self.split_name, self.subset_portion)
        else:
            return self.split_name


def parse_xml(xml_string: str, filter_valid=True):
    root = ET.fromstring(xml_string)
    # Parse question
    facet_ids = set()
    # Parse reference answers
    reference_answer_list = []
    for ra in root.find('referenceAnswers'):
        valid_token_id = set()
        tokens = []
        # Parse tokens
        for t in ra.find('tokens'):
            token = Token(id=t.attrib['id'], text=t.text)
            tokens.append(token)
            valid_token_id.add(token.id)

        # Parse facets
        facets = []
        for f in ra.find('facets'):
            facet = Facet(
                id=f.attrib['id'],
                govNum=f.attrib['govNum'],
                modNum=f.attrib['modNum'],
                reltn=f.attrib['reltn'],
                govText=f.attrib['govText'],
                modText=f.attrib['modText']
            )
            if 'childProp1Facet1' in f.attrib:
                facet.childProp1Facet1 = f.attrib['childProp1Facet1']
            if 'childProp2Facet1' in f.attrib:
                facet.childProp2Facet1 = f.attrib['childProp2Facet1']

            include = not filter_valid or facet.is_valid()

            if include:
                facet_ids.add(facet.id)
                facets.append(facet)

        reference_answer = ReferenceAnswer(
            id=ra.attrib['id'], text=ra.attrib['text'], tokens=tokens, facets=facets)

    question = Question(
        id=root.attrib['id'],
        module=root.attrib['module'],
        # test_set=root.attrib['testSet'],
        question_text=root.find('questionText').text,
        reference_answer=reference_answer
    )

    # Parse student answers
    for sa in root.find('studentAnswers'):
        student_answer = StudentAnswer(id=sa.attrib['id'], answer_text=sa.attrib['answerText'],
                                       accuracy=sa.attrib['accuracy'])

        # Parse facet entailments
        for fe in sa:
            fe = FacetEntailment(facet_id=fe.attrib['facetID'], label=fe.attrib['label'])

            if not filter_valid:
                f_include = True
            else:
                f_include = fe.facet_id in facet_ids and fe.label in [Expressed, Unaddressed]

            if f_include:
                student_answer.facet_entailments.append(fe)

        question.student_answers.append(student_answer)
    return question


def load_scientsbank_split(split, filter_valid=True) -> List[Question]:
    if isinstance(split, SplitSpec):
        split_name = split.split_name
        select_subset = split.use_subset
    else:
        split_name = split
        select_subset = False

    pte_dir = get_pte_dataset(split_name)

    question_list: List[Question] = []
    for file_path in get_dir_files(pte_dir):
        if file_path.endswith(".xml"):
            base_name = os.path.basename(file_path)
            f = open(file_path, "r")
            question: Question = parse_xml(f.read(), filter_valid)
            question_list.append(question)

    question_list.sort(key=lambda q: q.id)
    if not question_list:
        raise ValueError()
    if select_subset:
        sel_size = int(len(question_list) * split.subset_portion)
        if sel_size == 0:
            sel_size = 1
        assert sel_size > 0
        question_list = question_list[:sel_size]
    return question_list


sci_ents_test_split_list = ["test-unseen-answers", "test-unseen-domains", "test-unseen-questions"]
sci_ents_all_splits = ["train"] + sci_ents_test_split_list


def get_split_spec(split_name):
    if split_name == "train_sub":
        return SplitSpec("train", use_subset=True, subset_portion=0.1)
    elif split_name == "train_first":
        return SplitSpec("train", use_subset=True, subset_portion=0.0001)
    elif split_name == "train":
        return SplitSpec("train", use_subset=False)
    else:
        if split_name == "test1":
            split_name = "test-unseen-answers"
        elif split_name == "test2":
            split_name = "test-unseen-domains"
        elif split_name == "test3":
            split_name = "test-unseen-questions"
        return SplitSpec(split_name, use_subset=False)