from data_generator.job_runner import JobRunner, sydney_working_dir
from tlm.data_gen.run_sero_gen_wiki import SeroGen
from tlm.ukp.data_gen.clueweb12_13B import CluewebLMWorker

if __name__ == "__main__":
    target_seq_len = (128 - 2) * 24
    top_k = 10000
    generator = SeroGen(target_seq_len, False)
    JobRunner(sydney_working_dir, 1208, "sero_clueweb12_13B", lambda x: CluewebLMWorker(x, generator)).start()


