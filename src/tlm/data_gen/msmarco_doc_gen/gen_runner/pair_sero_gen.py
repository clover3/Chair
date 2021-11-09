from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import MultiWindow
from tlm.data_gen.msmarco_doc_gen.gen_worker import FirstPassagePairGenerator
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource(split)
    total_sequence_length = 512 * 4
    src_window_size = 512

    encoder = MultiWindow(src_window_size, total_sequence_length)

    generator = FirstPassagePairGenerator(resource, encoder, total_sequence_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)

    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_pair_512_4".format(split), factory)
    runner.start()
