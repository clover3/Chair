import os
import sys

from arg.counter_arg_retrieval.build_dataset.run3.swtt.nli_common import EncoderForNLI
from bert_api.predictor import FloatPredictor, PredictorWrap, Predictor
from bert_api.swtt.segmentwise_tokenized_text import SegmentwiseTokenizedText
from bert_api.swtt.swtt_scorer import DocumentScorerSWTT
from bert_api.swtt.swtt_scorer_def import SWTTScorerOutput
from bert_api.swtt.window_enum_policy import WindowEnumPolicyMinPop
from data_generator.NLI.nli_info import corpus_dir
from data_generator.tokenize_helper import TokenizedText
from data_generator.tokenizer_wo_tf import get_tokenizer


def read_data():
    filename = os.path.join(corpus_dir, "dev_matched.tsv")
    label_list = ["entailment", "neutral", "contradiction",]
    for idx, line in enumerate(open(filename, "rb")):
        if idx == 0: continue  # skip header
        line = line.strip().decode("utf-8")
        split_line = line.split("\t")
        # Works for both splits even though dev has some extra human labels.
        s1, s2 = split_line[8:10]
        l = label_list.index(split_line[-1])
        yield (s1, s2), l



def main():
    model_path = sys.argv[1]
    inner_predictor = Predictor(model_path, 3, 300)
    predictor: FloatPredictor = PredictorWrap(inner_predictor, lambda x: x[2])
    document_scorer = DocumentScorerSWTT(predictor, EncoderForNLI, 300)
    tokenizer = get_tokenizer()
    n_target = 10
    n_inst = 0
    for (prem, hypo), label in read_data():
        segment = TokenizedText.from_text(prem, tokenizer)
        doc = SegmentwiseTokenizedText([segment])
        enum_policy = WindowEnumPolicyMinPop(30, 50)
        q_tokens = tokenizer.tokenize(hypo)
        score = document_scorer.score(q_tokens, doc, enum_policy.window_enum)
        document_scorer.pk.do_duty()
        output: SWTTScorerOutput = score.get()
        print(label, output.scores, prem, hypo)
        n_inst += 1
        if n_inst >= n_target:
            break



if __name__ == "__main__":
    main()
