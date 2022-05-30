from typing import Iterator

from dataset_specific.mnli.parsing_jobs.partition_specs import get_mnli_spacy_ps
from dataset_specific.mnli.parsing_jobs.run_spacy import NLIPairDataSpacy


def read_spacy_nli(split) -> Iterator[NLIPairDataSpacy]:
    pds = get_mnli_spacy_ps(split)
    return pds.read_pickles_as_itr()