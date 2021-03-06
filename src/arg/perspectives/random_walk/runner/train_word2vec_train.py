
import os

from arg.perspectives.random_walk.graph_embedding import train_word2vec
from arg.perspectives.random_walk.graph_embedding_train_worker import GraphEmbeddingTrainWorker
from cpath import cache_path
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        input_file = os.path.join(cache_path, "pc_train_paras_list_form.pickle")
        return GraphEmbeddingTrainWorker(out_dir, input_file,
                                         train_word2vec)

    runner = JobRunner(sydney_working_dir, 453-1, "pc_train_word2vec", worker_gen)
    runner.start()



