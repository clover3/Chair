import os
import pickle

from data_generator import job_runner
from data_generator.job_runner import JobRunner, sydney_working_dir
from data_generator.tokenizer_wo_tf import is_continuation
from tlm.ukp.data_gen.clueweb12_13B import get_all_file_name_list


class CluewebTokenWorker(job_runner.WorkerInterface):
    def __init__(self, out_path):
        self.out_dir = out_path
        self.token_path = "/mnt/nfs/work3/youngwookim/data/clueweb12-B13_tokens/"
        self.all_name = get_all_file_name_list()

    def load_tokens_by_job_id(self, job_id):
        d = {}
        st = job_id * 10
        ed = (job_id+1) * 10
        for file_name in self.all_name[st:ed]:
            path = os.path.join(self.token_path, file_name)
            try:
                data : dict= pickle.load(open(path, "rb"))
                d.update(data)
            except FileNotFoundError as e:
                print(e)
        print("Loaded {} docs for {}".format(len(d), job_id))
        return d

    def subword_to_word(self, subword_tokens):
        def join_subwords(sbword_list):
            if len(sbword_list) == 1:
                return sbword_list[0]
            else:
                return "".join(sbword_list[:1] + [s[2:] for s in sbword_list[1:]])

        cur_word = []
        token_output = []

        for subword in subword_tokens:
            if is_continuation(subword):
                cur_word.append(subword)
            else:
                if cur_word:
                    token_output.append(join_subwords(cur_word))

                cur_word = [subword]
        if cur_word:
            token_output.append(join_subwords(cur_word))

        return token_output

    def work(self, job_id):
        token_d = self.load_tokens_by_job_id(job_id)
        docs = token_d.values()

        corpus = []
        for doc in docs:
            for line in doc:
                out_line = self.subword_to_word(line)
                corpus.append(out_line)
        output_file = os.path.join(self.out_dir, str(job_id))
        pickle.dump(corpus, open(output_file, "wb"))


if __name__ == "__main__":
    JobRunner(sydney_working_dir, 1208, "clueweb12_13B_word_tokens", CluewebTokenWorker).start()


