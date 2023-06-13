import dataclasses
import json
from json import JSONEncoder

from dataset_specific.scientsbank.pte_data_types import PTEPredictionPerQuestion, PTEPredictionPerStudentAnswer, \
    PTEPredictionPerFacet


class EnhancedJSONEncoder(JSONEncoder):
    def default(self, o):
        if dataclasses.is_dataclass(o):
            return dataclasses.asdict(o)
        return super().default(o)


def save_to_file(obj, filename):
    with open(filename, 'w') as outfile:
        json.dump(obj, outfile, cls=EnhancedJSONEncoder)


def load_from_file(filename):
    with open(filename, 'r') as infile:
        data = json.load(infile)

    def from_dict(data_class, data):
        if isinstance(data, list):
            return [from_dict(data_class, i) for i in data]
        else:
            return data_class(**{k: from_dict(t, v) for k, v, t in data_class.__annotations__.items()})

    return from_dict(PTEPredictionPerQuestion, data)


def dev_pred_parsing():
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

    save_to_file(pte_prediction, 'prediction.json')
    loaded_prediction = load_from_file('prediction.json')
    print(loaded_prediction)