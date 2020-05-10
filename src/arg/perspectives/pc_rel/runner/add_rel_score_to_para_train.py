from arg.perspectives.pc_rel.pc_rel_filter_to_paragraph_worker import PCRelFilterToParagraphWorker
from data_generator.job_runner import JobRunner, sydney_working_dir

if __name__ == "__main__":

    def worker_factory(out_dir):
        rel_info_pickle_path = "pc_rel_with_cpid"
        return PCRelFilterToParagraphWorker("pc_rel_tfrecord", rel_info_pickle_path, out_dir)

    runner = JobRunner(sydney_working_dir, 605, "rel_score_to_para_train", worker_factory)
    runner.start()


