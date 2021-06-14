
from data_generator.job_runner import JobRunner
from epath import job_man_dir
from tlm.data_gen.adhoc_sent_tokenize import FromTextEncoderShortSent
from tlm.data_gen.msmarco_doc_gen.gen_worker import MMDWorker
from tlm.data_gen.msmarco_doc_gen.gen_worker_sent_level import PredictionGen
from tlm.data_gen.msmarco_doc_gen.processed_resource import ProcessedResourcePredict





if __name__ == "__main__":
    for split in ["test", "dev"]:
        resource = ProcessedResourcePredict(split)
        max_seq_length = 256
        document_encoder = FromTextEncoderShortSent(max_seq_length, 60, None, 100)
        generator = PredictionGen(resource, document_encoder, max_seq_length)

        def factory(out_dir):
            return MMDWorker(resource.query_group, generator, out_dir)

        runner = JobRunner(job_man_dir, len(resource.query_group)-1, "MMD_{}_short_sent_split3".format(split), factory)
        runner.start()
