from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from tlm.data_gen.adhoc_sent_tokenize import SeroFromTextEncoder
from tlm.data_gen.msmarco_doc_gen.gen_worker_sent_level import PairwiseGenFromText
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc

if __name__ == "__main__":
    split = "train"
    max_segments = 4
    resource = ProcessedResource10doc(split)
    total_sequence_length = 512 * max_segments
    src_window_size = 512
    basic_encoder = SeroFromTextEncoder(src_window_size, total_sequence_length,
                                        random_short=True,
                                        max_seg_per_doc=max_segments)

    generator = PairwiseGenFromText(resource, basic_encoder, total_sequence_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunnerS(job_man_dir, len(resource.query_group)-1, "MMD_pair_512_4_B".format(split), factory)
    runner.start()
