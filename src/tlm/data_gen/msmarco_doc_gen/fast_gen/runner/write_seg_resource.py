import os
import pickle
from typing import Iterator

from data_generator.job_runner import WorkerInterface, JobRunner
from epath import job_man_dir
from misc_lib import exist_or_mkdir
from tlm.data_gen.msmarco_doc_gen.fast_gen.make_seg_resource import SegResourceMaker
from tlm.data_gen.msmarco_doc_gen.fast_gen.seg_resource import SRPerQuery
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc, ProcessedResourcePredict


class SegResourceWriterWorker(WorkerInterface):
    def __init__(self, query_group, seg_resource_maker, out_dir):
        self.out_dir = out_dir
        self.query_group = query_group
        self.seg_resource_maker = seg_resource_maker

    def work(self, job_id):
        qids = self.query_group[job_id]
        sr_itr: Iterator[SRPerQuery] = self.seg_resource_maker.generate(qids)
        for sr in sr_itr:
            out_path = os.path.join(self.out_dir, sr.qid)
            pickle.dump(sr, open(out_path, "wb"))


def do_for_split(split):
    if split == "train":
        resource = ProcessedResource10doc("train")
    else:
        resource = ProcessedResourcePredict(split)

    srm = SegResourceMaker(resource, 512, max_seg_per_doc=40)

    def factory(out_dir):
        return SegResourceWriterWorker(resource.query_group, srm, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1,
                       "seg_resource_{}".format(split), factory)
    runner.start()


def main():
    for split in ["dev", "train", "test"]:
        do_for_split(split)


if __name__ == "__main__":
    main()
