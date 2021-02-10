import collections
from typing import Iterable
from typing import Iterator

from arg.qck.decl import QCKQuery, QCKCandidate, get_light_qckquery, get_light_qckcandidate
from tf_util.record_writer_wrap import write_records_w_encode_fn
from tlm.data_gen.classification_common import QueryDocInstance, \
    ClassificationInstanceWDataID, encode_classification_instance_w_data_id
from tlm.data_gen.robust_gen.robust_generator_common import RobustGenCommon
from tlm.data_gen.robust_gen.seg_lib.segment_composer import IDBasedEncoder


class RobustGenLight(RobustGenCommon):
    def __init__(self,
                 encoder: IDBasedEncoder,
                 max_seq_length, include_all_judged):
        super(RobustGenLight, self).__init__()
        self.encoder = encoder
        self.max_seq_length = max_seq_length
        self.include_all_judged = include_all_judged

    def generate(self, query_list, data_id_manager) -> Iterator[QueryDocInstance]:
        neg_k = self.neg_k
        for query_id in query_list:
            if query_id not in self.judgement:
                continue

            qck_query = QCKQuery(query_id, "")
            judgement = self.judgement[query_id]
            ranked_list = self.galago_rank[query_id]
            ranked_list = ranked_list[:neg_k]

            target_docs = set()
            docs_in_ranked_list = [e.doc_id for e in ranked_list]
            target_docs.update(docs_in_ranked_list)

            if self.include_all_judged:
                docs_in_judgements = judgement.keys()
                target_docs.update(docs_in_judgements)

            print("Total of {} docs".format(len(target_docs)))
            for doc_id in target_docs:
                for tas in self.encoder.encode(query_id, doc_id):
                    label = 1 if doc_id in judgement and judgement[doc_id] > 0 else 0
                    candidate = QCKCandidate(doc_id, "")
                    info = {
                        'query': get_light_qckquery(qck_query),
                        'candidate': get_light_qckcandidate(candidate),
                    }
                    data_id = data_id_manager.assign(info)
                    inst = ClassificationInstanceWDataID.make_from_tas(tas, label, data_id)
                    yield inst

    def write(self, insts: Iterable[ClassificationInstanceWDataID], out_path: str):
        def encode_fn(inst: ClassificationInstanceWDataID) -> collections.OrderedDict :
            return encode_classification_instance_w_data_id(self.tokenizer, self.max_seq_length, inst)
        write_records_w_encode_fn(out_path, encode_fn, insts, 0)


class RobustTrainGenLight(RobustGenLight):
    def __init__(self, encoder, max_seq_length):
        super(RobustTrainGenLight, self).__init__(encoder, max_seq_length, True)


class RobustPredictGenLight(RobustGenLight):
    def __init__(self, encoder, max_seq_length):
        super(RobustPredictGenLight, self).__init__(encoder, max_seq_length, False)

