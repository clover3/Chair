import numpy as np

from tf_util.enum_features import load_record_v2
from tlm.data_gen.feature_to_text import take
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford


def get_correctness(filename, file_path):
    itr = load_record_v2(file_path)
    data = EstimatorPredictionViewerGosford(filename)

    correctness = []
    for entry in data:
        features = itr.__next__()

        input_ids = entry.get_vector("input_ids")
        input_ids2 = take(features["input_ids"])
        assert np.all(input_ids == input_ids2)
        label = take(features["label_ids"])[0]
        logits = entry.get_vector("logits")
        pred = np.argmax(logits)

        if pred == label:
            correctness.append(1)
        else:
            correctness.append(0)
    return correctness