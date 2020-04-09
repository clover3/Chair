import os
import pickle
from typing import Set, List, Dict

from arg.pf_common.ParagraphTFRecordWorker import ParagraphTFRecordWorker
from arg.pf_common.base import ParagraphFeature, DPID
from arg.ukp.data_loader import load_all_data, UkpDataPoint
from list_lib import lfilter


class UKPSplitInfo:
    def __init__(self):
        train, val = load_all_data()
        self.train = train
        self.val = val

    def get_dp_id_set(self, split, blind_topic) -> Set[DPID]:
        s = set()
        if split == 'train':
            train_data: Dict[str, List[UkpDataPoint]] = self.train
            for topic in train_data.keys():
                if topic != blind_topic:
                    for dp in train_data[topic]:
                        s.add(DPID(str(dp.id)))
        elif split == 'dev':
            dev_data = self.val
            for dp in dev_data[blind_topic]:
                s.add(DPID(str(dp.id)))
        return s


class UKPParagraphTFRecordWorker(ParagraphTFRecordWorker):
    def __init__(self, input_job_name: str,
                 split: str,
                 blind_topic: str,
                 out_dir: str):
        super(UKPParagraphTFRecordWorker, self).__init__(input_job_name, out_dir)
        self.dp_id_set: Set[str] = UKPSplitInfo().get_dp_id_set(split, blind_topic)

    def work(self, job_id):
        features: List[ParagraphFeature] = pickle.load(open(os.path.join(self.input_dir, str(job_id)), "rb"))

        def include(f: ParagraphFeature) -> bool:
            return f.datapoint.id in self.dp_id_set

        features: List[ParagraphFeature] = lfilter(include, features)

        if features:
            self.write(features, job_id)
        else:
            print("No features")