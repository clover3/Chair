from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_datagen import MultiWindow
from tlm.data_gen.msmarco_doc_gen.gen_worker import PointwiseGen
from tlm.data_gen.msmarco_doc_gen.mmd_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResource10doc

if __name__ == "__main__":
    split = "train"
    resource = ProcessedResource10doc(split)
    total_sequence_length = 512 * 4
    src_window_size = 512
    basic_encoder = MultiWindow(src_window_size, total_sequence_length)

    generator = PointwiseGen(resource, basic_encoder, total_sequence_length)

    def factory(out_dir):
        return MMDWorker(resource.query_group, generator, out_dir)
    runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_sero_train_10doc".format(split), factory)
    runner.start()
