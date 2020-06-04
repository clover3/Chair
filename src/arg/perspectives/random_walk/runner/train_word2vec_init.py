import os

from arg.perspectives.random_walk.graph_embedding import train_word2vec
from arg.perspectives.random_walk.graph_embedding_train_worker import GraphEmbeddingTrainWorker
from cpath import cache_path
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        input_file = os.path.join(cache_path, "pc_dev_paras_top_100_list_form.pickle")
        return GraphEmbeddingTrainWorker(out_dir, input_file,
                                         train_word2vec)

    runner = JobRunner(sydney_working_dir, 112, "pc_dev_word2vec_2", worker_gen)
    runner.start()


