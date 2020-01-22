from data_generator.job_runner import sydney_working_dir, JobRunner
from tlm.data_gen.base import UnmaskedPairGen
from tlm.data_gen.lm_worker import WikiLMWorker


class LongOnlyGen(UnmaskedPairGen):
    def __init__(self):
        super(LongOnlyGen, self).__init__()
        self.threshold = 3024 * 0.5

    def filter_long_docs(self, documents):
        for doc in documents:
            tokens_len = 0
            for segment in doc:
                tokens_len += len(segment)

            if tokens_len > self.threshold:
                yield doc

    def create_instances_from_documents(self, documents):
        return super(LongOnlyGen, self).create_instances_from_documents(self.filter_long_docs(documents))

    def write_instances(self, insts, out_path):
        self.write_instance_to_example_files(insts, [out_path])


if __name__ == "__main__":
    working_dir = sydney_working_dir
    runner = JobRunner(working_dir, 1000, "wiki_long_only", lambda out_path: WikiLMWorker(out_path, LongOnlyGen))
    runner.start()
