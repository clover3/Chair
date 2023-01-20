from typing import List, Dict

from contradiction.mnli_ex.nli_ex_common import NLIExEntry, nli_ex_entry_to_sent_token_label
from contradiction.token_tagging.acc_eval.defs import SentTokenLabel
from data_generator.data_parser.esnli import parse_judgement

target_class_d = {
    'conflict': 'contradiction',
    'match': 'entailment',
    'mismatch': 'neutral',
}

def load_esnli(split, tag_type) -> List[NLIExEntry]:
    entries = parse_judgement(split)
    def convert(e: Dict):
        return NLIExEntry(
            e['pairID'],
            e['Sentence1'],
            e['Sentence2'],
            e['indice1'],
            e['indice2']
        )
    target_label = target_class_d[tag_type]
    target_entries = [e for e in entries if e['gold_label'] == target_label]
    return list(map(convert, target_entries))



def load_esnli_binary_label(split, tag_type) -> List[SentTokenLabel]:
    entries = load_esnli(split, tag_type)
    output: List[SentTokenLabel] = []
    for e in entries:
        output.extend(nli_ex_entry_to_sent_token_label(e, tag_type))
    return output

