import os
import pickle
from typing import List, Dict, Tuple

from arg.perspectives.doc_value_viewer.calculate_doc_score import calculate_score
from arg.qck.doc_value_calculator import QCKOutEntry, logit_to_score_softmax, DocValueParts
from cpath import output_path
from estimator_helper.output_reader import join_prediction_with_info
from list_lib import lmap
from taskman_client.file_watching_job_runner import FileWatchingJobRunner


def load_baseline() -> Dict[Tuple[str, str], float]:
    # 2. Load baseline scores
    tf_record_dir = os.environ["tf_record_dir"]
    baseline_info_file_path = os.path.join(tf_record_dir, "baseline_ext.info")
    info = pickle.load(open(baseline_info_file_path, "rb"))
    out_dir = os.path.join(output_path, "cppnc_auto")
    pred_path = os.path.join(out_dir, "baseline_ext.score")
    predictions: List[Dict] = join_prediction_with_info(pred_path, info, ["logits"], False)
    out_entries: List[QCKOutEntry] = lmap(QCKOutEntry.from_dict, predictions)

    baseline_d: Dict[Tuple[str, str], float] = {}
    for e in out_entries:
        key = e.query.query_id, e.candidate.id
        score = logit_to_score_softmax(e.logits)
        baseline_d[key] = score

    return baseline_d


class ScoreSummarizer:
    def __init__(self):
        self.request_dir = os.environ["request_dir"]
        self.tf_record_dir = os.environ["tf_record_dir"]
        info_path = os.path.join(self.request_dir, "score_summarizer_job_info.json")
        self.save_dir = os.path.join(output_path, "cppnc_auto")
        score_save_path_format = os.path.join(self.save_dir, "{}.score")
        self.job_runner = FileWatchingJobRunner(score_save_path_format,
                                                info_path,
                                                self.summarize_score_and_save,
                              "score summarize")
        self.baseline_score: Dict[Tuple[str, str], float] = load_baseline()
        print("")
        print("  [ ScoreSummarizer ]")
        print()

    def file_watch_daemon(self):
        self.job_runner.start()

    def summarize_score_and_save(self, job_id: int):
        info_save_path = os.path.join(self.tf_record_dir, "{}.info".format(job_id))
        id_to_info = pickle.load(open(info_save_path, "rb"))
        score_save_path = os.path.join(self.save_dir, "{}.score".format(job_id))
        # calculate score for each kdp
        doc_score_parts: List[DocValueParts] = calculate_score(id_to_info, score_save_path, self.baseline_score)
        summary_save_path = os.path.join(self.save_dir, "{}.summary".format(job_id))
        pickle.dump(doc_score_parts, open(summary_save_path, "wb"))


if __name__ == "__main__":
    worker = ScoreSummarizer()
    worker.file_watch_daemon()