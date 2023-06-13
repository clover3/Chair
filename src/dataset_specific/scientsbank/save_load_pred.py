import dataclasses
import json
from dataclasses import dataclass, field
from typing import List, Any, Type, TypeVar
from json import JSONEncoder

from dataset_specific.scientsbank.pte_data_types import PTEPredictionPerQuestion, PTEPredictionPerStudentAnswer, \
    PTEPredictionPerFacet

T = TypeVar('T')


class EnhancedJSONEncoder(JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def save_pte_preds_to_file(obj: List[PTEPredictionPerQuestion], save_path):
    with open(save_path, 'w') as outfile:
        for item in obj:
            s = json.dumps(item, cls=EnhancedJSONEncoder)
            outfile.write(s + "\n")


def from_dict(data_class: Type[T], data: Any) -> T:
    if isinstance(data, list):
        return [from_dict(data_class.__args__[0], i) for i in data]
    elif isinstance(data, dict):
        return data_class(**{k: from_dict(t, data[k]) for k, t in data_class.__annotations__.items()})
    else:
        return data


def load_pte_preds_from_file(filename) -> List[PTEPredictionPerQuestion]:
    with open(filename, 'r') as infile:
        item_list = []
        for line in infile:
            data = json.loads(line)
            item = from_dict(PTEPredictionPerQuestion, data)
            item_list.append(item)
        return item_list


def main():
    # Example usage
    pte_prediction = PTEPredictionPerQuestion(
        id='question1',
        per_student_answer_list=[
            PTEPredictionPerStudentAnswer(
                id='answer1',
                facet_pred=[
                    PTEPredictionPerFacet('facet1', 0.9, True),
                    PTEPredictionPerFacet('facet2', 0.1, False)
                ]
            ),
            PTEPredictionPerStudentAnswer(
                id='answer2',
                facet_pred=[
                    PTEPredictionPerFacet('facet3', 0.9, True),
                    PTEPredictionPerFacet('facet4', 0.1, False)
                ]
            )
        ]
    )
    save_pte_preds_to_file([pte_prediction, pte_prediction], 'prediction.json')
    loaded_prediction = load_pte_preds_from_file('prediction.json')
    print(loaded_prediction)


if __name__ == "__main__":
    main()
