
from cache import load_from_pickle
from cpath import pjoin, output_path
from tlm.alt_emb.clef_ehealth.prediction_to_ranked_list import prediction_to_ranked_list
from tlm.estimator_prediction_viewer import EstimatorPredictionViewerGosford


subdir_root = pjoin(output_path, "eHealth")

def main():
    prediction_name = "eHealth_pred"
    pred_data = EstimatorPredictionViewerGosford(prediction_name)
    info = load_from_pickle("eHealth_test_info")
    out_path = pjoin(subdir_root , "eHealth_list.txt")
    prediction_to_ranked_list(pred_data, info, out_path)


def bert_baseline():
    prediction_name = "eHealth_bert_freeze"
    pred_data = EstimatorPredictionViewerGosford(prediction_name)
    info = load_from_pickle("eHealth_test_info")
    out_path = pjoin(subdir_root , "bert_baseline.txt")
    prediction_to_ranked_list(pred_data, info, out_path)


def bert_baseline2():
    prediction_name = "eHealth_pred.bert2"
    pred_data = EstimatorPredictionViewerGosford(prediction_name)
    info = load_from_pickle("eHealth_test_info")
    out_path = pjoin(subdir_root , "bert_baseline2.txt")
    prediction_to_ranked_list(pred_data, info, out_path)


def clueweb12_13A():
    prediction_name = "eHealth_pred.clueweb_12_13A"
    pred_data = EstimatorPredictionViewerGosford(prediction_name)
    info = load_from_pickle("eHealth_test_info")
    out_path = pjoin(subdir_root , "clef1_C.txt")
    prediction_to_ranked_list(pred_data, info, out_path)


def bert_baseline_repeat():
    info = load_from_pickle("eHealth_test_info")
    for i in [3,4,5]:
        prediction_name = "eHealth_bert_freeze_{}".format(i)
        pred_data = EstimatorPredictionViewerGosford(prediction_name)
        out_path = pjoin(subdir_root, "bert_baseline_{}.txt".format(i))
        prediction_to_ranked_list(pred_data, info, out_path)


if __name__ == "__main__":
    bert_baseline_repeat()
