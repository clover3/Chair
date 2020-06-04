import os
import pickle
from typing import Dict, List

from arg.perspectives.random_walk.graph_embedding import GraphEmbeddingTrainer
from cache import load_pickle_from
from data_generator import job_runner


class GraphEmbeddingTrainWorker(job_runner.WorkerInterface):
    def __init__(self,
                 out_path,
                 input_file,
                 train_fn: GraphEmbeddingTrainer,
                 ):
        self.out_dir = out_path
        self.corpus_d: Dict[int, List[List[str]]] = load_pickle_from(input_file)
        self.key_list = list(self.corpus_d.keys())
        self.key_list.sort()
        self.train_fn: GraphEmbeddingTrainer = train_fn

    def work(self, job_id):
        key = self.key_list[job_id]
        corpus: List[List[str]] = self.corpus_d[key]
        model = self.train_fn(corpus)
        save_path = os.path.join(self.out_dir, str(job_id))
        output = key, model
        pickle.dump(output, open(save_path, "wb"))