from abc import ABC, abstractmethod


class DocQueryEncoderInterface(ABC):
    @abstractmethod
    def encode(self, query_id, doc_id):
        # returns tokens, segmend_ids
        pass
