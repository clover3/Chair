import tensorflow_datasets as tfds

from epath import job_man_dir
from job_manager.job_runner_with_server import JobRunnerS
from misc_lib import ceil_divide
from trainer_v2.epr.segmentation import SegmentWorker


def segment_nli_common(dataset, dataset_name, item_per_job, split):
    print(f"segment_nli_common({dataset_name}, {split})")

    def get_worker(out_dir):
        worker = SegmentWorker(out_dir,
                               item_per_job,
                               dataset)
        return worker

    num_item = len(dataset)
    n_job = ceil_divide(num_item, item_per_job)
    job_name = f"{dataset_name}_{split}_tokenize"
    job_runner = JobRunnerS(job_man_dir, n_job, job_name, get_worker)
    job_runner.start()


def main():
    item_per_job = 10000
    for dataset_name in ["snli", "multi_nli"]:
        all_dataset = tfds.load(name=dataset_name)
        for split in all_dataset.keys():
            dataset = all_dataset[split]
            segment_nli_common(dataset, dataset_name, item_per_job, split)


if __name__ == "__main__":
    main()