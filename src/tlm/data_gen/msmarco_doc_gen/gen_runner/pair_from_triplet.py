import os
from collections import OrderedDict

from cpath import output_path
from data_generator.job_runner import JobRunner
from data_generator.tokenizer_wo_tf import get_tokenizer
from epath import job_man_dir
from galagos.types import Query
from misc_lib import DataIDManager
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.adhoc_datagen import LeadingN
from tlm.data_gen.classification_common import PairedInstance
from tlm.data_gen.msmarco_doc_gen.triplet.reader import Doc, read_triplet
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from tlm.data_gen.pairwise_common import combine_features


class FirstPassagePairGenerator:
    def __init__(self, max_seq_length):
        self.tokenizer = get_tokenizer()
        self.encoder = LeadingN(max_seq_length, 1)
        self.max_seq_length = max_seq_length

    def triplet_to_paired_instance(self, query, doc1: Doc, doc2: Doc, data_id: int)\
            -> PairedInstance:
        q_tokens = self.tokenizer.tokenize(query.text)

        def get_doc_tokens(doc: Doc):
            text = doc.title + "\n" + doc.content
            return self.tokenizer.tokenize(text)

        doc1_tokens = get_doc_tokens(doc1)
        doc2_tokens = get_doc_tokens(doc2)

        encoder = self.encoder
        insts1: List[Tuple[List, List]] = encoder.encode(q_tokens, doc1_tokens)
        insts2: List[Tuple[List, List]] = encoder.encode(q_tokens, doc2_tokens)
        assert len(insts1) == 1
        assert len(insts2) == 1

        tokens_seg1, seg_ids1 = insts1[0]
        tokens_seg2, seg_ids2 = insts2[0]
        inst = PairedInstance(tokens_seg1, seg_ids1, tokens_seg2, seg_ids2, data_id)
        return inst

    def generate(self, triplet_itr: Iterator[Tuple[Query, Doc, Doc]]):
        data_id_manager = DataIDManager()
        for q, d1, d2 in triplet_itr:
            data_id = data_id_manager.assign({})
            inst = self.triplet_to_paired_instance(q, d1, d2, data_id)
            yield inst

    def write(self, insts: Iterable[PairedInstance], out_path, length=0):
        def encode_fn(inst: PairedInstance) -> OrderedDict:
            return combine_features(inst.tokens1, inst.seg_ids1, inst.tokens2, inst.seg_ids2,
                                    self.tokenizer, self.max_seq_length)
        return write_records_w_encode_fn(out_path, encode_fn, insts, length)


class TripletWorker:
    def __init__(self, input_path_format, out_dir: str):
        self.generator = FirstPassagePairGenerator(512)
        self.out_dir = out_dir
        self.input_path_format = input_path_format

    def work(self, job_id):
        input_path = self.input_path_format.format(job_id)
        triplet_itr: Iterator[Tuple[Query, Doc, Doc]] = read_triplet(input_path)
        insts = self.generator.generate(triplet_itr)
        save_path = os.path.join(self.out_dir, str(job_id))
        self.generator.write(insts, save_path)


def do_at_once():
    generator = FirstPassagePairGenerator(512)
    input_path = "/mnt/nfs/work3/youngwookim/data/msmarco/triples.tsv"
    triplet_itr: Iterator[Tuple[Query, Doc, Doc]] = read_triplet(input_path)
    insts = generator.generate(triplet_itr)
    save_path = os.path.join(output_path, "mmd", "train_triplet.tfrecord")
    num_item = 360000
    generator.write(insts, save_path, num_item)


def main():
    input_path_format= "/mnt/nfs/work3/youngwookim/data/msmarco/triple_pieces/x{0:04}"

    def factory(out_dir):
        return TripletWorker(input_path_format, out_dir)

    num_jobs = 360
    runner = JobRunner(job_man_dir, num_jobs-1, "MMD_pair_triplet", factory)
    runner.start()


if __name__ == "__main__":
    main()
