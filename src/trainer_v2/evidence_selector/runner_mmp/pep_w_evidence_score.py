import pickle
from typing import List, Tuple, Any
import numpy as np

# Assuming the necessary imports for types like EvidencePair2 if they exist
from data_generator2.segmented_enc.es_common.es_two_seg_common import EvidencePair2, EvidencePair3


def load_predictions(path: str) -> List[Tuple[EvidencePair2, np.array]]:
    """
    Load the pickled predictions from a given path.

    Args:
    - path (str): The path to the pickled file.

    Returns:
    - List[Tuple[EvidencePair2, Any]]: A list of tuples where each tuple contains an instance of `EvidencePair2`
                                       and the corresponding output (prediction).
    """
    with open(path, 'rb') as file:
        loaded_data = pickle.load(file)
    return loaded_data


def encode_evidence_pair(evidence_pair: EvidencePair3):
    for part_no in [0, 1]:
        query_side: List[str] = evidence_pair.query_like_segment.get(part_no)
        evidence_all: List[str] = evidence_pair.evidence_like_segment.tokens


# Load the data
def main():
    predict_save_path = NotImplemented
    output_rows_loaded: List[Tuple[EvidencePair2, np.array]]\
        = load_predictions(predict_save_path)

    # TODO Generate data



if __name__ == "__main__":
    main()