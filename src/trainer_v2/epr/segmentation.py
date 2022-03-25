import os
from typing import List

import spacy

from cache import save_list_to_jsonl
from contradiction.ie_align.srl.spacy_segmentation import spacy_segment
from data_generator.job_runner import WorkerInterface
from misc_lib import TimeEstimator


class SegmentWorker(WorkerInterface):
    def __init__(self, out_dir, item_per_job, dataset):
        self.nlp = spacy.load("en_core_web_sm")
        self.out_dir = out_dir
        self.dataset = dataset
        self.item_per_job = item_per_job


    def work(self, job_id):
        def fetch_text(string_tensor) -> str:
            return string_tensor.numpy().decode("utf-8")

        item_per_job = self.item_per_job
        st = job_id * item_per_job
        ed = st + item_per_job
        iter = self.dataset.skip(st).take(item_per_job)
        print("{} items in dataset".format(len(iter)))
        ticker = TimeEstimator(len(iter))
        out_item_list = []
        for item in iter:
            new_d = {}
            for seg_name in ["premise", "hypothesis"]:
                new_d[seg_name] = self.segment(fetch_text(item[seg_name]))
            new_d['label'] = int(item['label'].numpy())
            ticker.tick()
            out_item_list.append(new_d)

        save_path = os.path.join(self.out_dir, str(job_id))
        save_list_to_jsonl(out_item_list, save_path)

    def segment(self, text) -> List[str]:
        spacy_tokens = self.nlp(text)
        span_list = spacy_segment(spacy_tokens)
        return list(map(str, span_list))


