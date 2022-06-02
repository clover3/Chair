import spacy

from data_generator.segmented_enc.sent_split_by_spacy import split_spacy_tokens
from dataset_specific.mnli.mnli_reader import MNLIReader
from dataset_specific.mnli.parsing_jobs.partition_specs import get_mnli_partition_spec
from job_manager.job_runner3 import IteratorWorkerSpec, run_iterator_to_pickle_worker
from misc_lib import tprint


class SpacySplitWorker(IteratorWorkerSpec):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def work(self, data_itr):
        output = []
        for e in data_itr:
            text = e.hypothesis
            text_tuple = split_spacy_tokens(self.nlp(e.hypothesis))
            output.append((text, text_tuple))
        tprint("job done for {} items".format(len(output)))
        return output


def main():
    worker = SpacySplitWorker()
    reader = MNLIReader()
    split = "train"
    data_itr = lambda :reader.load_split(split)
    ps = get_mnli_partition_spec(split)
    name = "mnli_spacy_split_{}".format(split)
    run_iterator_to_pickle_worker(ps, data_itr, worker, name)


if __name__ == "__main__":
    main()