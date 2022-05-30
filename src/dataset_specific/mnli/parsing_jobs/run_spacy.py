import itertools
from typing import List, NamedTuple

import spacy

from cache import save_to_pickle
from data_generator.NLI.enlidef import nli_label_list
from dataset_specific.mnli.mnli_reader import MNLIReader, NLIPairData
from dataset_specific.mnli.parsing_jobs.partition_specs import get_mnli_partition_spec
from job_manager.job_runner3 import IteratorWorkerSpec, run_iterator_to_pickle_worker
from misc_lib import TELI, tprint


class NLIPairDataSpacy(NamedTuple):
    premise: str
    hypothesis: str
    label: str
    data_id: str
    p_spacy_tokens: List
    h_spacy_tokens: List

    def get_label_as_int(self):
        return nli_label_list.index(self.label)

    @classmethod
    def from_nli_pair_data(cls, e: NLIPairData, nlp):
        return NLIPairDataSpacy(e.premise, e.hypothesis, e.label, e.data_id,
                                nlp(e.premise), nlp(e.hypothesis))


class SpacyWorker(IteratorWorkerSpec):
    def __init__(self):
        self.nlp = spacy.load("en_core_web_sm")

    def work(self, data_itr):
        output = []
        for e in data_itr:
            output.append(NLIPairDataSpacy.from_nli_pair_data(e, self.nlp))
        tprint("job done for {} items".format(len(output)))
        return output


def check_working_time():
    split = "train"
    nlp = spacy.load("en_core_web_sm")
    reader = MNLIReader()
    data_itr = reader.load_split(split)
    output = []
    for e in TELI(data_itr, reader.get_data_size(split)):
        output.append(NLIPairDataSpacy.from_nli_pair_data(e, nlp))

    save_to_pickle(output, "mnli_spacy_{}".format(split))


def main():
    worker = SpacyWorker()
    reader = MNLIReader()
    split = "train"
    data_itr = lambda :reader.load_split(split)
    ps = get_mnli_partition_spec(split)
    name = "mnli_spacy_tokenize_{}".format(split)
    run_iterator_to_pickle_worker(ps, data_itr, worker, name)


def debug():
    reader = MNLIReader()
    split = "train"
    data_itr_fn = lambda :reader.load_split(split)
    job_id = 379
    def show_count(job_id):
        num_record_per_job = 1000
        st = num_record_per_job * job_id
        ed = st + num_record_per_job
        data_itr = data_itr_fn()
        itr = itertools.islice(iter(data_itr), st, ed)
        index = 0
        for _ in itr:
            index += 1
        print(index)

    show_count(379)
    show_count(350)



if __name__ == "__main__":
    # debug()
    main()