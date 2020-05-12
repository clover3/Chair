from arg.perspectives.random_walk.biased_random_walk_worker import BiasedRandomWalkWorker
from cache import load_from_pickle
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":
    def worker_gen(out_dir):
        prob_score_d = load_from_pickle("pc_neg_word_prob_train")
        input_dir = "/mnt/nfs/work3/youngwookim/data/bert_tf/pc_train_co_occur"
        return BiasedRandomWalkWorker(out_dir, input_dir, prob_score_d)

    runner = JobRunner(sydney_working_dir, 453, "bias_random_walk_train_neg", worker_gen)
    runner.start()

