from typing import List, Dict

from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import MultiWindow
from tlm.data_gen.adhoc_sent_tokenize import SeroFromTextEncoder
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.gen_worker_sent_level import PointwiseGenFromText
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)
    max_segments = 4
    total_sequence_length = 512 * 4
    src_window_size = 512
    basic_encoder = SeroFromTextEncoder(src_window_size, total_sequence_length,
                                        random_short=True,
                                        max_seg_per_doc=max_segments)

    generator = PointwiseGenFromText(resource, basic_encoder, total_sequence_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)
    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_sero_train_A", factory)
    runner.start()
